from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
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