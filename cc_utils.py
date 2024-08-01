from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from hardware_model.device import Device
import time
from dataclasses import dataclass

def get_model_params(llm: str):
    if llm == 'gpt3':
        d_model = 12288
        d_ffn = 12288 * 4
        n_heads = 96
        n_kv_heads = 96
        activation = 'gelu'
        n_layers = 96
    elif llm == 'llama3-70B':
        d_model = 8192
        d_ffn = 28672
        n_heads = 64
        n_kv_heads = 8
        activation = 'silu'
        n_layers = 80
    elif llm == 'llama3-405B':
        d_model = 16384
        d_ffn = 30720
        n_heads = 128
        n_kv_heads = 8
        activation = 'silu'
        n_layers = 126
    else:
        raise ValueError(f"llm: {llm} not supported")
    return d_model, d_ffn, n_heads, n_kv_heads, activation, n_layers

@dataclass
class Latency:
    qkv_proj: float
    q_mul_k: float
    a_mul_v: float
    o_matmul: float
    ffn_matmul1: float
    ffn_matmul2: float
    ffn_matmul3: float
    hadamard_mul: float
    softmax: float
    layernorm: float
    activation: float
    allreduce: float
    total: float

def sim_prefill(llm: str, hw_name: str, device_count: int, input_len: int,  bs: int, 
                compile_mode: str = 'heuristic-GPU') -> Latency:

    hw_specs = read_architecture_template(f'./configs/{hw_name}.json')
    system = template_to_system(hw_specs)
    d_model, d_ffn, n_heads, n_kv_heads, activation, n_layers = get_model_params(llm)

    prefill_model = TransformerBlockInitComputationTP(
        d_model=d_model,
        n_heads=n_heads,
        device_count=device_count,
        data_type=data_type_dict["fp16"],
        activation=activation,
        d_ffn=d_ffn,
        n_kv_heads=n_kv_heads
    )

    print(f"Simmulating prefill model on {hw_name} with input_len: {input_len}")
    start = time.time()
    _ = prefill_model(Tensor([bs, input_len, d_model], data_type_dict["fp16"]))
    prefill_latency = prefill_model.compile_and_simulate(system, compile_mode)
    qkv_latency, q_mul_k_latency, a_mul_v_latency, h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency, h3_matmul3_latency, swi_mul_latency, softmax_latency, layernorm_latency, _, act_latency, allreduce_latency, _ = prefill_model.simluate_log.split(", ")
    latency_breakdown = Latency(
        qkv_proj=float(qkv_latency) * n_layers,
        q_mul_k=float(q_mul_k_latency) * n_layers,
        a_mul_v=float(a_mul_v_latency) * n_layers,
        o_matmul=float(h_matmul0_latency) * n_layers,
        ffn_matmul1=float(h1_matmul1_latency) * n_layers,
        ffn_matmul2=float(h2_matmul2_latency) * n_layers,
        ffn_matmul3=float(h3_matmul3_latency) * n_layers,
        hadamard_mul=float(swi_mul_latency) * n_layers,
        softmax=float(softmax_latency) * n_layers,
        layernorm=float(layernorm_latency) * 2 * n_layers,
        activation=float(act_latency) * n_layers,
        allreduce=float(allreduce_latency) * 2 * n_layers,
        total=prefill_latency * n_layers
    )
    throughput = bs * input_len / latency_breakdown.total
    end = time.time()

    print(f'Finished in {int(end - start)} seconds. Prefill latency: {prefill_latency} seconds, throughput: {throughput:.1f} tokens/second.') 

    return latency_breakdown

def sim_decode(llm: str, hw_name: str, device_count: int, input_len: int, output_len: int, bs: int,
               compile_mode: str = 'heuristic-GPU') -> Latency:
    hw_specs = read_architecture_template(f'./configs/{hw_name}.json')
    system = template_to_system(hw_specs)
    d_model, d_ffn, n_heads, n_kv_heads, activation, n_layers = get_model_params(llm)

    decode_model = TransformerBlockAutoRegressionTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=device_count,
            data_type=data_type_dict["fp16"],
            activation=activation,
            d_ffn=d_ffn,
            n_kv_heads=n_kv_heads
        )

    print(f"Simmulating decode model on {hw_name} with input_len: {input_len}, output_len: {output_len}")
    start = time.time()
    seq_len = input_len + output_len // 2
    _ = decode_model(Tensor([bs, 1, d_model], data_type_dict["fp16"]), seq_len)
    decode_latency = decode_model.compile_and_simulate(system, compile_mode)
    qkv_latency, q_mul_k_latency, a_mul_v_latency, h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency, h3_matmul3_latency, swi_mul_latency, softmax_latency, layernorm_latency, _, act_latency, allreduce_latency, _ = decode_model.simluate_log.split(", ")
    latency_breakdown = Latency(
        qkv_proj=float(qkv_latency) * n_layers * output_len,
        q_mul_k=float(q_mul_k_latency) * n_layers * output_len,
        a_mul_v=float(a_mul_v_latency) * n_layers * output_len,
        o_matmul=float(h_matmul0_latency) * n_layers * output_len,
        ffn_matmul1=float(h1_matmul1_latency) * n_layers * output_len,
        ffn_matmul2=float(h2_matmul2_latency) * n_layers * output_len,
        ffn_matmul3=float(h3_matmul3_latency) * n_layers * output_len,
        hadamard_mul=float(swi_mul_latency) * n_layers * output_len,
        softmax=float(softmax_latency) * n_layers * output_len,
        layernorm=float(layernorm_latency) * 2 * n_layers * output_len,
        activation=float(act_latency) * n_layers * output_len,
        allreduce=float(allreduce_latency) * 2 * n_layers * output_len,
        total=decode_latency * n_layers * output_len
    )
    throughput = bs * output_len / latency_breakdown.total
    end = time.time()

    print(f'Finished in {int(end - start)} seconds. Decode latency: {decode_latency} seconds, throughput: {throughput:.1f} tokens/second.')

    return latency_breakdown

def get_matmul_latency(model: TransformerBlockInitComputationTP, device: Device, mappings: dict):
    q_latency = model.Q_proj.run_given_mapping(device, mappings['q_proj']) + device.compute_module.overhead.matmul
    k_layency = model.K_proj.run_given_mapping(device, mappings['k_proj']) + device.compute_module.overhead.matmul
    qkv_latency = q_latency + 2 * k_layency
    h_matmul0_latency = model.H_matmul0.run_given_mapping(device, mappings['h_matmul0']) + device.compute_module.overhead.matmul
    h_matmul1_latency = model.H_matmul1.run_given_mapping(device, mappings['h_matmul1']) + device.compute_module.overhead.matmul
    h_matmul2_latency = model.H_matmul2.run_given_mapping(device, mappings['h_matmul2']) + device.compute_module.overhead.matmul
    return qkv_latency, h_matmul0_latency, h_matmul1_latency, h_matmul2_latency

def get_batched_matmul_latency(model: TransformerBlockInitComputationTP, device: Device, mappings: dict):
    q_mul_k_latency = model.Q_mul_K.run_given_mapping(device, mappings['q_mul_k'][0], mappings['q_mul_k'][1]) + device.compute_module.overhead.matmul
    a_mul_v_latency = model.A_mul_V.run_given_mapping(device, mappings['a_mul_v'][0], mappings['a_mul_v'][1]) + device.compute_module.overhead.matmul
    return q_mul_k_latency, a_mul_v_latency

def sim_prefill_fast(llm: str, hw_name: str, device_count: int, input_lens: list, 
                     batch_sizes: list, use_flash_attn: bool = False) -> dict:

    compile_mode = "heuristic-GPU-fast"
    hw_specs = read_architecture_template(f'./configs/{hw_name}.json')
    system = template_to_system(hw_specs)

    d_model, d_ffn, n_heads, n_kv_heads, activation, n_layers = get_model_params(llm)

    prefill_model = TransformerBlockInitComputationTP(
        d_model=d_model,
        n_heads=n_heads,
        device_count=device_count,
        data_type=data_type_dict["fp16"],
        activation=activation,
        d_ffn=d_ffn,
        n_kv_heads=n_kv_heads,
        use_flash_attn=use_flash_attn
    )

    results = dict()

    device = system.device
    interconnect = system.interconnect
    matmul_mappings = dict()
    for i, input_len in enumerate(input_lens):
        batched_matmul_mappings = dict() # q_mul_k, a_mul_v
        for j, bs in enumerate(batch_sizes):
            key = f"{input_len}_{bs}"
            print(f"Simmulating prefill model on {hw_name} with input_len: {input_len} at batch_size: {bs}")
            _ = prefill_model(Tensor([bs, input_len, d_model], data_type_dict["fp16"]))

            if i == 0 and j == 0:
                prefill_latency = prefill_model.compile_and_simulate(system, compile_mode)
                qkv_latency, q_mul_k_latency, a_mul_v_latency, h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency, h3_matmul3_latency, swi_mul_latency, softmax_latency, layernorm_latency, _, act_latency, allreduce_latency, _ = prefill_model.simluate_log.split(", ")

                matmul_mappings['q_proj'] = prefill_model.Q_proj.best_mapping
                matmul_mappings['k_proj'] = prefill_model.K_proj.best_mapping
                matmul_mappings['h_matmul0'] = prefill_model.H_matmul0.best_mapping
                matmul_mappings['h_matmul1'] = prefill_model.H_matmul1.best_mapping
                matmul_mappings['h_matmul2'] = prefill_model.H_matmul2.best_mapping

                if not use_flash_attn:
                    batched_matmul_mappings['q_mul_k'] = [prefill_model.Q_mul_K.best_mapping1, prefill_model.Q_mul_K.best_mapping2]
                    batched_matmul_mappings['a_mul_v'] = [prefill_model.A_mul_V.best_mapping1, prefill_model.A_mul_V.best_mapping2]
            else:
                if use_flash_attn:
                    prefill_model.flash_attn.compile_and_simulate(device)
                    q_mul_k_latency = prefill_model.flash_attn.sim_q_mul_k_lat + device.compute_module.overhead.matmul
                    a_mul_v_latency = prefill_model.flash_attn.sim_a_mul_v_lat + device.compute_module.overhead.matmul
                else:
                    if j == 0:
                        # Rerun the batched matmuls
                        start = time.time()
                        a_mul_v_latency = prefill_model.A_mul_V.compile_and_simulate(system.device, compile_mode=compile_mode)
                        q_mul_k_latency = prefill_model.Q_mul_K.compile_and_simulate(system.device, compile_mode=compile_mode)
                        batched_matmul_mappings['q_mul_k'] = [prefill_model.Q_mul_K.best_mapping1, prefill_model.Q_mul_K.best_mapping2]
                        batched_matmul_mappings['a_mul_v'] = [prefill_model.A_mul_V.best_mapping1, prefill_model.A_mul_V.best_mapping2]
                        end = time.time()
                        if end - start > 30:
                            print(f"Rerun batched matmuls for input_len: {input_len}, bs: {bs} took {end - start} seconds.")
                    else:
                        # Reuse the batched matmul mappings
                        q_mul_k_latency, a_mul_v_latency = get_batched_matmul_latency(prefill_model, device, batched_matmul_mappings)

                # Reuse the matmul mappings
                qkv_latency, h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency = get_matmul_latency(prefill_model, device, matmul_mappings)
                # Get other latencies
                if use_flash_attn:
                    softmax_latency = prefill_model.flash_attn.sim_softmax_lat + device.compute_module.overhead.softmax
                else:
                    softmax_latency = prefill_model.A_softmax.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.softmax
                layernorm_latency = prefill_model.layer_norm0.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.layernorm

                if activation == 'gelu':
                    act_latency = prefill_model.H_gelu.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.gelu
                    swi_mul_latency = 0
                    h3_matmul3_latency = 0
                elif activation == 'silu':
                    act_latency = prefill_model.H_silu.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.gelu
                    swi_mul_latency = prefill_model.swi_mul.compile_and_simulate(device, compile_mode) 
                    h3_matmul3_latency = h1_matmul1_latency

                if device_count > 1:
                    allreduce_latency = prefill_model.allreduce_mha.simulate(interconnect)
                else:
                    allreduce_latency = 0
                
                prefill_latency = qkv_latency + q_mul_k_latency + a_mul_v_latency + h_matmul0_latency + h1_matmul1_latency + h2_matmul2_latency + h3_matmul3_latency + swi_mul_latency + softmax_latency + layernorm_latency * 2 + act_latency + allreduce_latency * 2
            latency_breakdown = Latency(
                qkv_proj=float(qkv_latency) * n_layers,
                q_mul_k=float(q_mul_k_latency) * n_layers,
                a_mul_v=float(a_mul_v_latency) * n_layers,
                o_matmul=float(h_matmul0_latency) * n_layers,
                ffn_matmul1=float(h1_matmul1_latency) * n_layers,
                ffn_matmul2=float(h2_matmul2_latency) * n_layers,
                ffn_matmul3=float(h3_matmul3_latency) * n_layers,
                hadamard_mul=float(swi_mul_latency) * n_layers,
                softmax=float(softmax_latency) * n_layers,
                layernorm=float(layernorm_latency) * 2 * n_layers,
                activation=float(act_latency) * n_layers,
                allreduce=float(allreduce_latency) * 2 * n_layers,
                total=prefill_latency * n_layers
            )
            results[key] = latency_breakdown
            throughput = bs * input_len / latency_breakdown.total
            print(f'Input len: {input_len}, bs: {bs}, Prefill latency: {prefill_latency} seconds, throughput: {throughput:.1f} tokens/second.') 

    return results

def sim_decode_fast(llm: str, hw_name: str, device_count: int, input_lens: list, output_lens: list,
                     batch_sizes: list, use_flash_attn: bool = False) -> dict:

    compile_mode = "heuristic-GPU-fast"
    hw_specs = read_architecture_template(f'./configs/{hw_name}.json')
    system = template_to_system(hw_specs)

    d_model, d_ffn, n_heads, n_kv_heads, activation, n_layers = get_model_params(llm)

    decode_model = TransformerBlockAutoRegressionTP(
        d_model=d_model,
        n_heads=n_heads,
        device_count=device_count,
        data_type=data_type_dict["fp16"],
        activation=activation,
        d_ffn=d_ffn,
        n_kv_heads=n_kv_heads,
        use_flash_attn=use_flash_attn
    ) 

    results = dict()

    device = system.device
    interconnect = system.interconnect
    matmul_mappings = dict()
    for i, input_len in enumerate(input_lens):
        for k, output_len in enumerate(output_lens):
            batched_matmul_mappings = dict() # q_mul_k, a_mul_v
            for j, bs in enumerate(batch_sizes):
                key = f"{input_len}_{output_len}_{bs}"
                print(f"Simmulating decode model on {hw_name} with input_len: {input_len}, output_len: {output_len} at batch_size: {bs}")
                _ = decode_model(Tensor([bs, 1, d_model], data_type_dict["fp16"]), input_len + output_len // 2)

                if i == 0 and k == 0 and j == 0:
                    decode_latency = decode_model.compile_and_simulate(system, compile_mode)
                    qkv_latency, q_mul_k_latency, a_mul_v_latency, h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency, h3_matmul3_latency, swi_mul_latency, softmax_latency, layernorm_latency, _, act_latency, allreduce_latency, _ = decode_model.simluate_log.split(", ")

                    matmul_mappings['q_proj'] = decode_model.Q_proj.best_mapping
                    matmul_mappings['k_proj'] = decode_model.K_proj.best_mapping
                    matmul_mappings['h_matmul0'] = decode_model.H_matmul0.best_mapping
                    matmul_mappings['h_matmul1'] = decode_model.H_matmul1.best_mapping
                    matmul_mappings['h_matmul2'] = decode_model.H_matmul2.best_mapping

                    if not use_flash_attn:
                        batched_matmul_mappings['q_mul_k'] = [decode_model.Q_mul_K.best_mapping1, decode_model.Q_mul_K.best_mapping2]
                        batched_matmul_mappings['a_mul_v'] = [decode_model.A_mul_V.best_mapping1, decode_model.A_mul_V.best_mapping2]
                else:
                    if use_flash_attn:
                        decode_model.flash_attn.compile_and_simulate(device)
                        q_mul_k_latency = decode_model.flash_attn.sim_q_mul_k_lat + device.compute_module.overhead.matmul
                        a_mul_v_latency = decode_model.flash_attn.sim_a_mul_v_lat + device.compute_module.overhead.matmul
                    else:
                        if j == 0:
                            # Rerun the batched matmuls
                            start = time.time()
                            a_mul_v_latency = decode_model.A_mul_V.compile_and_simulate(system.device, compile_mode=compile_mode)
                            q_mul_k_latency = decode_model.Q_mul_K.compile_and_simulate(system.device, compile_mode=compile_mode)
                            batched_matmul_mappings['q_mul_k'] = [decode_model.Q_mul_K.best_mapping1, decode_model.Q_mul_K.best_mapping2]
                            batched_matmul_mappings['a_mul_v'] = [decode_model.A_mul_V.best_mapping1, decode_model.A_mul_V.best_mapping2]
                            end = time.time()
                            if end - start > 30:
                                print(f"Rerun batched matmuls for input_len: {input_len}, bs: {bs} took {end - start} seconds.")
                        else:
                            # Reuse the batched matmul mappings
                            q_mul_k_latency, a_mul_v_latency = get_batched_matmul_latency(decode_model, device, batched_matmul_mappings)

                    # Reuse the matmul mappings
                    qkv_latency, h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency = get_matmul_latency(decode_model, device, matmul_mappings)
                    # Get other latencies
                    if use_flash_attn:
                        softmax_latency = decode_model.flash_attn.sim_softmax_lat + device.compute_module.overhead.softmax
                    else:
                        softmax_latency = decode_model.A_softmax.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.softmax
                    layernorm_latency = decode_model.layer_norm0.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.layernorm

                    if activation == 'gelu':
                        act_latency = decode_model.H_gelu.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.gelu
                        swi_mul_latency = 0
                        h3_matmul3_latency = 0
                    elif activation == 'silu':
                        act_latency = decode_model.H_silu.compile_and_simulate(device, compile_mode) + device.compute_module.overhead.gelu
                        swi_mul_latency = decode_model.swi_mul.compile_and_simulate(device, compile_mode) 
                        h3_matmul3_latency = h1_matmul1_latency

                    if device_count > 1:
                        allreduce_latency = decode_model.allreduce_mha.simulate(interconnect)
                    else:
                        allreduce_latency = 0

                    decode_latency = qkv_latency + q_mul_k_latency + a_mul_v_latency + h_matmul0_latency + h1_matmul1_latency + h2_matmul2_latency + h3_matmul3_latency + swi_mul_latency + softmax_latency + layernorm_latency * 2 + act_latency + allreduce_latency * 2
                latency_breakdown = Latency(
                    qkv_proj=float(qkv_latency) * n_layers * output_len,
                    q_mul_k=float(q_mul_k_latency) * n_layers * output_len,
                    a_mul_v=float(a_mul_v_latency) * n_layers * output_len,
                    o_matmul=float(h_matmul0_latency) * n_layers * output_len,
                    ffn_matmul1=float(h1_matmul1_latency) * n_layers * output_len,
                    ffn_matmul2=float(h2_matmul2_latency) * n_layers * output_len,
                    ffn_matmul3=float(h3_matmul3_latency) * n_layers * output_len,
                    hadamard_mul=float(swi_mul_latency) * n_layers * output_len,
                    softmax=float(softmax_latency) * n_layers * output_len,
                    layernorm=float(layernorm_latency) * 2 * n_layers * output_len,
                    activation=float(act_latency) * n_layers * output_len,
                    allreduce=float(allreduce_latency) * 2 * n_layers * output_len,
                    total=decode_latency * n_layers * output_len
                )
                results[key] = latency_breakdown
                throughput = bs * output_len / latency_breakdown.total
                print(f'Input len: {input_len}, output len: {output_len}, bs: {bs}, Decode latency: {decode_latency} seconds, throughput: {throughput:.1f} tokens/second.')

    return results