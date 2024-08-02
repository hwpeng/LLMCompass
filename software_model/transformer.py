from software_model.operators import (
    Operator,
    Reshape,
    Concat,
    Transpose,
    Repeat,
)
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.silu import SiLU
from software_model.mul import Mul
from software_model.flashattention import FlashAttention


from software_model.utils import Tensor, DataType
from software_model.communication_primitives import AllReduceMultiPCB
from math import ceil
from typing import List, Optional
from hardware_model.system import System


class TransformerBlockInitComputationTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType, activation: str = "gelu", n_kv_heads: Optional[int] = None, d_ffn: Optional[int] = None, use_flash_attn: bool = False, use_flash_attn_xcel: bool = False, flash_attn_br: int = -1, flash_attn_bc: int = -1):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        self.activation = activation
        if n_kv_heads is None:
            self.n_kv_heads = n_heads
        else:
            self.n_kv_heads = n_kv_heads
        if d_ffn is None:
            self.d_ffn = d_model * 4
        else:
            self.d_ffn = d_ffn
        self.use_flash_attn = use_flash_attn
        self.use_flash_attn_xcel = use_flash_attn_xcel
        self.flash_attn_br = flash_attn_br
        self.flash_attn_bc = flash_attn_bc
        # parameters per device
        d = d_model
        d_h = d // n_heads
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d_h * self.n_kv_heads // device_count], data_type)
        self.Wv = Tensor([d, d_h * self.n_kv_heads // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, self.d_ffn // device_count], data_type)
        self.W2 = Tensor([self.d_ffn // device_count, d], data_type)
        if self.activation == "silu":
            # (swish(xw1) * (xw3)) * w3
            self.W3 = Tensor([d, self.d_ffn / device_count], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.K_repeat = Repeat(data_type)
        self.V_repeat = Repeat(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_silu = SiLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        if self.activation == "silu":
            self.swi_mul = Mul(data_type)
            self.H_matmul3 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)
        self.flash_attn = FlashAttention(data_type, self.use_flash_attn_xcel, self.flash_attn_br, self.flash_attn_bc)

    def __call__(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        kv_h = self.n_kv_heads

        # multi-head attention
        Q = self.Q_proj(X, self.Wq)  # [b, s, d / dev_cnt]
        assert Q.shape == [b, s, d // dev_cnt]
        K = self.K_proj(X, self.Wk)  # [b, s, d_h * n_kv_heads / dev_cnt]
        V = self.V_proj(X, self.Wv)  # [b, s, d_h * n_kv_heads / dev_cnt]
        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        K = self.K_reshape(K, [b, s, kv_h // dev_cnt, d_h])
        V = self.V_reshape(V, [b, s, kv_h // dev_cnt, d_h])
        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert Q_T.shape == [b, h // dev_cnt, s, d_h]
        K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, kv_h / dev_cnt, d_h, s]
        assert K_T.shape == [b, kv_h // dev_cnt, d_h, s]
        K_R = self.K_repeat(K_T, 1, h // kv_h) # [b, h / dev_cnt, d_h, s]
        assert K_R.shape == [b, h // dev_cnt, d_h, s]
        V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, kv_h / dev_cnt, s, d_h]
        assert V_T.shape == [b, kv_h // dev_cnt, s, d_h]
        V_R = self.V_repeat(V_T, 1, h // kv_h) # [b, h / dev_cnt, s, d_h]
        assert V_R.shape == [b, h // dev_cnt, s, d_h]
        if not self.use_flash_attn:
            A = self.Q_mul_K(Q_T, K_R)  # [b, h / dev_cnt, s, s]
            assert A.shape == [b, h // dev_cnt, s, s]
            A_prob = self.A_softmax(A)
            H = self.A_mul_V(A_prob, V_R)  #  [b, h / dev_cnt, s, d_h]
        else:
            H = self.flash_attn(Q_T, K_T, V_T, 0)
        # print(H.shape)
        # print([b, h // dev_cnt, s, d_h])
        assert H.shape == [b, h // dev_cnt, s, d_h]
        H = self.H_transpose(H, [0, 2, 1, 3])  #  [b, s, h / dev_cnt, d_h]
        assert H.shape == [b, s, h // dev_cnt, d_h]
        H = self.H_reshape(H, [b, s, d // dev_cnt])
        assert H.shape == [b, s, d // dev_cnt]
        H0 = self.H_matmul0(H, self.W0)  #  [b, s, d]
        assert H0.shape == [b, s, d]
        H0 = self.layer_norm0(H0)
        assert H0.shape == [b, s, d]
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)

        # feed-forward network
        H1 = self.H_matmul1(H0, self.W1)  # [b, s, d_ffn / dev_cnt]
        assert H1.shape == [b, s, self.d_ffn // dev_cnt]
        if self.activation == "gelu":
            H1 = self.H_gelu(H1)
        else:
            # swiglu
            # (swish(xw1) * (xw3)) * w2
            H1 = self.H_silu(H1)
            H3 = self.H_matmul3(H0, self.W3)
            H1 = self.swi_mul(H1, H3)
        H2 = self.H_matmul2(H1, self.W2)  #  [b, s, d]
        assert H2.shape == [b, s, d]
        H2 = self.layer_norm1(H2)
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)

        assert H2.shape == [b, s, d]
        return H2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        # pre-run, will grab components later as this is a fused kernel
        if self.use_flash_attn:
            self.flash_attn.roofline_model(device)

        qkv_latency = (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul +
            2 * (self.K_proj.roofline_model(device) + device.compute_module.overhead.matmul)
        )
        if not self.use_flash_attn:
            q_mul_k_latency = (
                self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
            )
            a_mul_v_latency = (
                self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
            )
        else:
            q_mul_k_latency = (
                self.flash_attn.q_mul_k_lat + device.compute_module.overhead.matmul
            )
            a_mul_v_latency = (
                self.flash_attn.a_mul_v_lat + device.compute_module.overhead.matmul
            )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        if self.activation == "silu":
            h3_matmul3_latency = h1_matmul1_latency
        else:
            h3_matmul3_latency = 0

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + h3_matmul3_latency
        )

        # normalization
        if not self.use_flash_attn:
            softmax_latency = (
                self.A_softmax.roofline_model(device)
                + device.compute_module.overhead.softmax
            )
        else:
            softmax_latency = (
                self.flash_attn.softmax_lat
                + device.compute_module.overhead.softmax
            )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        if self.activation == "gelu":
            # gelu
            act_latency = (
                self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
            )
            swi_mul_latency = 0
        else:
            act_latency = (
                self.H_silu.roofline_model(device) + device.compute_module.overhead.gelu
            )
            swi_mul_latency = (self.swi_mul.roofline_model(device))

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{act_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {h3_matmul3_latency}, {swi_mul_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {act_latency}, {allreduce_latency}, {allreduce_latency}"
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{act_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + act_latency
            + swi_mul_latency
            + allreduce_total_latency
        )
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        device = system.device
        interconnect = system.interconnect

        # pre-run, will grab components later as this is a fused kernel
        if self.use_flash_attn:
            print("simulating flash_attn")
            self.flash_attn.compile_and_simulate(device, compile_mode)

        # matmul
        print("simulating qkv")
        qkv_latency = (
            self.Q_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul +
            2 * (self.K_proj.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.matmul)
        )
        if not self.use_flash_attn:
            print("simulating q_mul_k")
            q_mul_k_latency = (
                self.Q_mul_K.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul
            )
            print("simulating a_mul_v")
            a_mul_v_latency = (
                self.A_mul_V.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul
            )
        else:
            q_mul_k_latency = (
                self.flash_attn.sim_q_mul_k_lat
                + device.compute_module.overhead.matmul
            )
            a_mul_v_latency = (
                self.flash_attn.sim_a_mul_v_lat
                + device.compute_module.overhead.matmul
            )
        print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.matmul
        )
        if self.activation == "silu":
            print("simulating h3_matmul3")
            h3_matmul3_latency = h1_matmul1_latency
        else:
            h3_matmul3_latency = 0
        print("finish matmul simulation")

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + h3_matmul3_latency
        )

        # normalization
        if not self.use_flash_attn:
            softmax_latency = (
                self.A_softmax.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.softmax
            )
        else:
            softmax_latency = (
                self.flash_attn.sim_softmax_lat
                + device.compute_module.overhead.softmax
            )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        if self.activation == "gelu":
            # gelu
            act_latency = (
                self.H_gelu.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.gelu
            )
            swi_mul_latency = 0
        else:
            act_latency = (
                self.H_silu.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.gelu
            )
            swi_mul_latency = (self.swi_mul.compile_and_simulate(device, compile_mode))

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + act_latency
            + allreduce_total_latency
            + swi_mul_latency
        )
        self.simluate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {h3_matmul3_latency}, {swi_mul_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {act_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def run_on_gpu(self):
        if self.use_flash_attn:
            raise RuntimeError("FlashAttention is not supported on GPU")

        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu() + 2 * self.K_proj.run_on_gpu()
        )
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )
        if self.activation == "silu":
            h3_matmul3_latency = (
                self.H_matmul3.run_on_gpu()  # - self.H_matmul3.gpu_kernel_launch_overhead()
            )
        else:
            h3_matmul3_latency = 0

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + h3_matmul3_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        if self.activation == "gelu":
            # gelu
            act_latency = (
                self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
            )
            swi_mul_latency = 0
        else:
            act_latency = (
                self.H_silu.run_on_gpu()  # - self.H_silu.gpu_kernel_launch_overhead()
            )
            swi_mul_latency = self.swi_mul.run_on_gpu()

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{act_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + act_latency
            + allreduce_total_latency
            + swi_mul_latency
        )
        return self.latency_on_gpu


class TransformerBlockAutoRegressionTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType, activation: str = "gelu", n_kv_heads: Optional[int] = None, d_ffn: Optional[int] = None, use_flash_attn: bool = False, use_flash_attn_xcel: bool = False, flash_attn_br: int = -1, flash_attn_bc: int = -1):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        self.activation = activation
        if n_kv_heads is None:
            self.n_kv_heads = n_heads
        else:
            self.n_kv_heads = n_kv_heads
        if d_ffn is None:
            self.d_ffn = d_model * 4
        else:
            self.d_ffn = d_ffn
        self.use_flash_attn = use_flash_attn
        self.use_flash_attn_xcel = use_flash_attn_xcel
        self.flash_attn_br = flash_attn_br
        self.flash_attn_bc = flash_attn_bc
        # parameters per device
        d = d_model
        d_h = d // n_heads
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d_h * self.n_kv_heads // device_count], data_type)
        self.Wv = Tensor([d, d_h * self.n_kv_heads // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, self.d_ffn // device_count], data_type)
        self.W2 = Tensor([self.d_ffn // device_count, d], data_type)
        if self.activation == "silu":
            # (swish(xw1) * (xw3)) * w3
            self.W3 = Tensor([d, self.d_ffn / device_count], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_repeat = Repeat(data_type)
        self.V_repeat = Repeat(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_silu = SiLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        if self.activation == "silu":
            self.swi_mul = Mul(data_type)
            self.H_matmul3 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)
        self.flash_attn = FlashAttention(data_type, self.use_flash_attn_xcel, self.flash_attn_br, self.flash_attn_bc)

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        kv_h = self.n_kv_heads

        # KV cache
        K_cache = Tensor([b, kv_h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, kv_h // dev_cnt, s, d_h], self.data_type)

        # multi-head attention
        q = self.Q_proj(x, self.Wq)  # [b, 1, d / dev_cnt]
        assert q.shape == [b, 1, d // dev_cnt]
        k = self.K_proj(x, self.Wk)  # [b, 1, d / dev_cnt]
        v = self.V_proj(x, self.Wv)  # [b, 1, d / dev_cnt]
        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, kv_h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, kv_h // dev_cnt, d_h])
        q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert q_T.shape == [b, h // dev_cnt, 1, d_h]
        k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, kv_h / dev_cnt, d_h, 1]
        assert k_T.shape == [b, kv_h // dev_cnt, d_h, 1]
        v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, kv_h / dev_cnt, 1, d_h]
        assert v_T.shape == [b, kv_h // dev_cnt, 1, d_h]
        if not self.use_flash_attn:
            K_T = self.K_concat(K_cache, k_T, 3)  # [b, kv_h / dev_cnt, d_h, s+1]
            assert K_T.shape == [b, kv_h // dev_cnt, d_h, s + 1]
            K_R = self.K_repeat(K_T, 1, h // kv_h)  # [b, h / dev_cnt, d_h, s+1]
            assert K_R.shape == [b, h // dev_cnt, d_h, s + 1]
            V_T = self.V_concat(V_cache, v_T, 2)  # [b, kv_h / dev_cnt, s+1, d_h]
            assert V_T.shape == [b, kv_h // dev_cnt, s + 1, d_h]
            V_R = self.V_repeat(V_T, 1, h // kv_h)  # [b, h / dev_cnt, s+1, d_h]
            assert V_R.shape == [b, h // dev_cnt, s + 1, d_h]
            a = self.Q_mul_K(q_T, K_R)  # [b, h / dev_cnt, 1, s+1]
            assert a.shape == [b, h // dev_cnt, 1, s + 1]
            a_prob = self.A_softmax(a)
            h0 = self.A_mul_V(a_prob, V_R)  #  [b, h / dev_cnt, 1, d_h]
        else:
            h0 = self.flash_attn(q_T, k_T, v_T, s)
        assert h0.shape == [b, h // dev_cnt, 1, d_h]
        h0 = self.H_transpose(h0, [0, 2, 1, 3])  #  [b, 1, h / dev_cnt, d_h]
        assert h0.shape == [b, 1, h // dev_cnt, d_h]
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        assert h0.shape == [b, 1, d // dev_cnt]
        h0 = self.H_matmul0(h0, self.W0)  #  [b, 1, d]
        assert h0.shape == [b, 1, d]
        h0 = self.layer_norm0(h0)
        assert h0.shape == [b, 1, d]
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)

        # feed-forward network

        # H1 = self.H_matmul1(H0, self.W1)  # [b, s, d_ffn / dev_cnt]
        # assert H1.shape == [b, s, self.d_ffn // dev_cnt]
        # if self.activation == "gelu":
        #     H1 = self.H_gelu(H1)
        # else:
        #     # swiglu
        #     # (swish(xw1) * (xw3)) * w2
        #     H1 = self.H_silu(H1)
        #     H3 = self.H_matmul3(H0, self.W3)
        #     H1 = self.swi_mul(H1, H3)

        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, d_ffn / dev_cnt]
        assert h1.shape == [b, 1, self.d_ffn // dev_cnt]
        if self.activation == "gelu":
            h1 = self.H_gelu(h1)
        else:
            # swiglu
            # (swish(xw1) * (xw3)) * w2
            h1 = self.H_silu(h1)
            h3 = self.H_matmul3(h0, self.W3)
            h1 = self.swi_mul(h1, h3)
        h2 = self.H_matmul2(h1, self.W2)  #  [b, 1, d]
        assert h2.shape == [b, 1, d]
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        assert h2.shape == [b, 1, d]
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + K_cache.size * K_cache.data_type.word_size
            + V_cache.size * V_cache.data_type.word_size
        )
        if self.activation == "silu":
            self.memory_requirement += self.W3.size * self.W3.data_type.word_size
        return h2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        # pre-run, will grab components later as this is a fused kernel
        if self.use_flash_attn:
            self.flash_attn.roofline_model(device)

        qkv_latency = (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul +
            2 * (self.K_proj.roofline_model(device) + device.compute_module.overhead.matmul)
        )
        if not self.use_flash_attn:
            q_mul_k_latency = (
                self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
            )
            a_mul_v_latency = (
                self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
            )
        else:
            q_mul_k_latency = (
                self.flash_attn.q_mul_k_lat + device.compute_module.overhead.matmul
            )
            a_mul_v_latency = (
                self.flash_attn.a_mul_v_lat + device.compute_module.overhead.matmul
            )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        if self.activation == "silu":
            h3_matmul3_latency = h1_matmul1_latency
        else:
            h3_matmul3_latency = 0

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + h3_matmul3_latency
        )

        # normalization
        if not self.use_flash_attn:
            softmax_latency = (
                self.A_softmax.roofline_model(device)
                + device.compute_module.overhead.softmax
            )
        else:
            softmax_latency = (
                self.flash_attn.softmax_lat
                + device.compute_module.overhead.softmax
            )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        if self.activation == "gelu":
            # gelu
            act_latency = (
                self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
            )
            swi_mul_latency = 0
        else:
            act_latency = (
                self.H_silu.roofline_model(device) + device.compute_module.overhead.gelu
            )
            swi_mul_latency = (self.swi_mul.roofline_model(device))

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{act_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {h3_matmul3_latency}, {swi_mul_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {act_latency}, {allreduce_latency}, {allreduce_latency}"
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{act_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + act_latency
            + swi_mul_latency
            + allreduce_total_latency
        )
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str):
        pcb = system.device
        interconnect = system.interconnect

        # pre-run, will grab components later as this is a fused kernel
        if self.use_flash_attn:
            # print("simulating flash_attn")
            self.flash_attn.compile_and_simulate(pcb, compile_mode)

        # matmul
        # print("simulating qkv")
        qkv_latency = (
            self.Q_proj.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul +
            2 * (self.K_proj.compile_and_simulate(pcb, compile_mode) + pcb.compute_module.overhead.matmul)
        )
        if not self.use_flash_attn:
            # print("simulating q_mul_k")
            q_mul_k_latency = (
                self.Q_mul_K.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.matmul
            )
            # print("simulating a_mul_v")
            a_mul_v_latency = (
                self.A_mul_V.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.matmul
            )
        else:
            q_mul_k_latency = (
                self.flash_attn.sim_q_mul_k_lat
                + pcb.compute_module.overhead.matmul
            )
            a_mul_v_latency = (
                self.flash_attn.sim_a_mul_v_lat
                + pcb.compute_module.overhead.matmul
            )
        # print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.matmul
        )
        if self.activation == "silu":
            # print("simulating h3_matmul3")
            h3_matmul3_latency = h1_matmul1_latency
        else:
            h3_matmul3_latency = 0

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + h3_matmul3_latency
        )

        # normalization
        if not self.use_flash_attn:
            softmax_latency = (
                self.A_softmax.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.softmax
            )
        else:
            softmax_latency = (
                self.flash_attn.sim_softmax_lat
                + pcb.compute_module.overhead.softmax
            )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        if self.activation == "gelu":
            # gelu
            act_latency = (
                self.H_gelu.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.gelu
            )
            swi_mul_latency = 0
        else:
            act_latency = (
                self.H_silu.compile_and_simulate(pcb, compile_mode)
                + pcb.compute_module.overhead.gelu
            )
            swi_mul_latency = self.swi_mul.compile_and_simulate(pcb, compile_mode)


        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + act_latency
            + allreduce_total_latency
            + swi_mul_latency
        )
        self.simluate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {h3_matmul3_latency}, {swi_mul_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {act_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def run_on_gpu(self):
        if self.use_flash_attn:
            raise RuntimeError("FlashAttention is not supported on GPU")

        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu() + 2 * self.K_proj.run_on_gpu()
        )
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )
        if self.activation == "silu":
            h3_matmul3_latency = (
                self.H_matmul3.run_on_gpu()  # - self.H_matmul3.gpu_kernel_launch_overhead()
            )
        else:
            h3_matmul3_latency = 0

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + h3_matmul3_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        if self.activation == "gelu":
            # gelu
            act_latency = (
                self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
            )
            swi_mul_latency = 0
        else:
            act_latency = (
                self.H_silu.run_on_gpu()  # - self.H_silu.gpu_kernel_launch_overhead()
            )
            swi_mul_latency = self.swi_mul.run_on_gpu()

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{act_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + act_latency
            + allreduce_total_latency
            + swi_mul_latency
        )
        return self.latency_on_gpu


class LLMInitComputationTP:
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        device_count,
    ) -> None:
        pass
