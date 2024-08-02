from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil
import numpy as np


class FlashAttention(Operator):
    def __init__(
        self, data_type: DataType, use_xcel: bool = False, br: int = -1, bc: int = -1
    ):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.input3_shape = None
        self.output_shape = None
        self.use_xcel = use_xcel
        self.xcel_freq = 1e9
        if use_xcel:
            self.br = 128
            self.bc = 128
        else:
            self.br = br
            self.bc = bc

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
        Bc = ceil(M / (4 * self.d_h)) if self.bc == -1 else self.bc
        Tc = ceil(N_kv / Bc)  # O( N_kv * d_h * M^-1 )

        # All other blocking
        Br = min(ceil(M / (4 * self.d_h)), self.d_h) if self.br == -1 else self.br
        Tr = ceil(N / Br)  # O( N * d_h * M^-1 )

        K_size = np.prod([self.bs, self.h, N_kv, self.d_h])
        V_size = np.prod([self.bs, self.h, N_kv, self.d_h])
        Q_size = np.prod([self.bs, self.h, N, self.d_h])
        O_size = np.prod([self.bs, self.h, N, self.d_h])
        l_size = N
        m_size = N

        q_mul_k_io_count = K_size + Tc * Q_size
        p_mul_v_io_count = V_size + Tc * O_size
        softmax_io_count = 2 * Tc * l_size + 2 * Tc * m_size + Tc * O_size
        self.io_count = q_mul_k_io_count + p_mul_v_io_count + softmax_io_count

        io_bw = min(
            pcb_module.io_module.bandwidth,
            pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq,
        )
        io_bound_latency = self.io_count / io_bw

        q_mul_k_flops = self.bs * self.h * (N * N_kv * self.d_h)
        p_mul_v_flops = self.bs * self.h * (N * N_kv * self.d_h)
        self.flop_count = q_mul_k_flops + p_mul_v_flops

        if not self.use_xcel:
            self.xcel_cycles = 0
            save_exps = False  # set to true to further reduce operation counting but would require more memory to store intermediate values
            exp_only = False  # set to true to use the older model

            if exp_only:
                softmax_exps = (
                    Br * Bc  # np.exp(sij - mij)
                    + Br  # np.exp(mi - m_new)
                    + Br  # np.exp(mij - m_new)
                    + (0 if save_exps else Br)  # np.exp(mi - m_new)
                    + (0 if save_exps else Br)  # np.exp(mij - m_new)
                )
                self.vec_count = (
                    self.bs
                    * self.h
                    * Tc
                    * Tr
                    * softmax_exps
                    * pcb_module.compute_module.core.vector_unit.flops_per_exp
                )

            else:
                mij_flops = Br * Bc
                pij_flops = (
                    Br
                    * Bc
                    * (pcb_module.compute_module.core.vector_unit.flops_per_exp + 1)
                )
                lij_flops = Br * Bc
                m_new_flops = Br
                exp_mi_flops = Br * (
                    pcb_module.compute_module.core.vector_unit.flops_per_exp + 1
                )
                exp_mi_li_flops = Br
                exp_mij_flops = Br * (
                    pcb_module.compute_module.core.vector_unit.flops_per_exp + 1
                )
                l_new_flops = 2 * Br
                o_flops = 4 * Br * self.d_h
                if not save_exps:
                    o_flops += (
                        exp_mi_flops + exp_mi_li_flops
                    ) + exp_mij_flops  # rematerialize
                self.vec_count = (
                    self.bs
                    * self.h
                    * Tc
                    * Tr
                    * sum(
                        [
                            mij_flops,
                            pij_flops,
                            lij_flops,
                            m_new_flops,
                            exp_mi_flops,
                            exp_mi_li_flops,
                            exp_mij_flops,
                            l_new_flops,
                            o_flops,
                        ]
                    )
                )
                print("FLASHATTENTION")
                print(f"{self.vec_count=}")
                print(f"{self.vec_count / pcb_module.compute_module.total_vector_flops=}")

        # use the xcel
        else:
            rowmax_cycles = 7
            exp_cycles = 1 + 20
            xcel_pipeline_depth = sum(
                [
                    rowmax_cycles,
                    exp_cycles,
                    2,  # +2 for exp_mi and exp_mij following up after sij
                ]
            )

            self.xcel_cycles = (
                self.bs * self.h * Tc * Tr * (Br + xcel_pipeline_depth)
            )

            self.vec_count = (
                self.bs
                * self.h
                * Tc
                * Tr
                * sum(
                    [
                        Br, # exp_mi_li
                        Br * Bc, # lij
                        4 * Br * self.d_h, # temp
                    ]
                )
            )
            print("XCEL FLASHATTN")
            print(f"{self.vec_count=}")
            print(f"{self.xcel_cycles=}")
            print(f"{self.vec_count / pcb_module.compute_module.total_vector_flops=}")
            print(f"{self.xcel_cycles / (self.xcel_freq * pcb_module.compute_module.core_count)=}")

        comp_bound_latency = (
            self.flop_count / pcb_module.compute_module.total_systolic_array_flops
            + self.vec_count / pcb_module.compute_module.total_vector_flops
            + self.xcel_cycles / (self.xcel_freq * pcb_module.compute_module.core_count)
        )

        if io_bound_latency > comp_bound_latency:
            print("IO BOUND")
            self.roofline_latency = io_bound_latency
            self.q_mul_k_lat = q_mul_k_io_count / io_bw
            self.a_mul_v_lat = p_mul_v_io_count / io_bw
            self.softmax_lat = softmax_io_count / io_bw
        else:
            print("COMP BOUND")
            self.roofline_latency = comp_bound_latency
            self.q_mul_k_lat = (
                q_mul_k_flops / pcb_module.compute_module.total_systolic_array_flops
            )
            self.a_mul_v_lat = (
                p_mul_v_flops / pcb_module.compute_module.total_systolic_array_flops
            )
            self.softmax_lat = (
                self.vec_count / pcb_module.compute_module.total_vector_flops
                + self.xcel_cycles / (self.xcel_freq * pcb_module.compute_module.core_count)
            )

        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str = ""):
        # TODO: Implement compile and simulate for FlashAttention
        self.latency = self.roofline_model(pcb_module)
        self.sim_q_mul_k_lat = self.q_mul_k_lat
        self.sim_a_mul_v_lat = self.a_mul_v_lat
        self.sim_softmax_lat = self.softmax_lat
        return self.latency
