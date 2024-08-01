from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil
import numpy as np


class FlashAttention(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.input3_shape = None
        self.output_shape = None

    def __call__(
        self, input1: Tensor, input2: Tensor, input3: Tensor, kv_len: int
    ) -> Tensor:
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        assert self.data_type == input3.data_type

        self.input1_shape = input1.shape  # q_T -> [ bs,    h,   s, d_h ]
        self.input2_shape = input2.shape  # k_T -> [ bs, kv_h, d_h, s ]
        self.input3_shape = input3.shape  # v_T -> [ bs, kv_h, s, d_h ]
        self.kv_len = kv_len

        (self.bs, self.h, self.s, self.d_h) = self.input1_shape
        self.kv_h = self.input2_shape[1]

        self.output_shape = [self.bs, self.h, self.s, self.d_h]
        output = Tensor(self.output_shape, self.data_type)

        return output

    def roofline_model(self, pcb_module: Device):
        M = pcb_module.compute_module.core.SRAM_size
        N = self.s
        N_kv = self.kv_len + self.s  # in prefill, kv_len == 0 therefore N_kv == N.
        # in decode, s == 1 so N_kv > N

        # KV blocking
        Bc = ceil(M / (4 * self.d_h))
        Tc = ceil(N_kv / Bc)  # O( N_kv * d_h * M^-1 )

        # All other blocking
        Br = min(ceil(M / (4 * self.d_h)), self.d_h)
        Tr = ceil(N / Br)  # O( N * d_h * M^-1 )

        K_size = np.prod([self.bs, self.h, N_kv, self.d_h])
        V_size = np.prod([self.bs, self.h, N_kv, self.d_h])
        Q_size = np.prod([self.bs, self.h, N, self.d_h])
        O_size = np.prod([self.bs, self.h, N, self.d_h])
        l_size = N
        m_size = N

        self.io_count = (
            K_size  # ld
            + V_size  # ld
            + Tc * Q_size  # ld
            + 2 * Tc * O_size  # ld/st
            + 2 * Tc * l_size  # ld/st
            + 2 * Tc * m_size  # ld/st
        )

        q_mul_k_io_count = K_size + Tc * Q_size
        p_mul_v_io_count = V_size + Tc * O_size
        softmax_io_count = 2 * Tc * l_size + 2 * Tc * m_size + Tc * O_size

        assert self.io_count == (q_mul_k_io_count + p_mul_v_io_count + softmax_io_count)

        io_bw = min(
            pcb_module.io_module.bandwidth,
            pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq,
        )

        io_bound_latency = self.io_count / io_bw

        q_mul_k_flops = self.bs * self.h * (N * N_kv * self.d_h)
        p_mul_v_flops = self.bs * self.h * (N * N_kv * self.d_h)
        softmax_exps = self.bs * self.h * (
            Tc
            * Tr
            * (
                Br * Bc  # np.exp(sij - mij)
                + Br  # np.exp(mi - m_new)
                + Br  # np.exp(mij - m_new)
                + Br  # np.exp(mi - m_new)
                + Br  # np.exp(mij - m_new)
            )
        )

        self.flop_count = q_mul_k_flops + p_mul_v_flops
        self.exp_count = softmax_exps

        comp_bound_latency = (
            self.flop_count / pcb_module.compute_module.total_systolic_array_flops
            + self.exp_count
            * pcb_module.compute_module.core.vector_unit.flops_per_exp
            / pcb_module.compute_module.total_vector_flops
        )

        if io_bound_latency > comp_bound_latency:
            self.roofline_latency = io_bound_latency
            self.q_mul_k_lat = q_mul_k_io_count / io_bw
            self.a_mul_v_lat = p_mul_v_io_count / io_bw
            self.softmax_lat = softmax_io_count / io_bw
        else:
            self.roofline_latency = comp_bound_latency
            self.q_mul_k_lat = q_mul_k_flops / pcb_module.compute_module.total_systolic_array_flops
            self.a_mul_v_lat = p_mul_v_flops / pcb_module.compute_module.total_systolic_array_flops
            self.softmax_lat = self.exp_count * pcb_module.compute_module.core.vector_unit.flops_per_exp / pcb_module.compute_module.total_vector_flops

        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str = ""):
        # TODO: Implement compile and simulate for FlashAttention
        self.latency = self.roofline_model(pcb_module)
        self.sim_q_mul_k_lat = self.q_mul_k_lat
        self.sim_a_mul_v_lat = self.a_mul_v_lat
        self.sim_softmax_lat = self.softmax_lat
        return self.latency
