# %%
from cc_utils import get_model_params, sim_prefill
# %%
# Setttings
hw_name = 'GH100'
device_count = 8
# For kernal partitioning and mapping, options are 
# "heuristic-GPU", "heuristic-TPU", "heuristic-our-throughput" ,"exhaustive", etc
compile_mode = "heuristic-GPU"
llm = 'llama3-70B'

batch_sizes = [1, 8, 16, 32]
input_lens = [4096, 32768, 65536, 131072]
d_model, d_ffn, n_heads, n_kv_heads, activation, n_layers = get_model_params(llm)

# %%
# Prefill
f = open('prefill_results.csv', mode='w')
f.write("input_len, bs, total, throughput, qkv_proj, q_mul_k, a_mul_v, o_matmul, ffn_matmul1, ffn_matmul2, ffn_matmul3, hadamard_mul, softmax, layernorm, activation, allreduce\n")
for bs in batch_sizes:
    print(f"Running prefill for batch size: {bs}")
    for input_len in input_lens:
        key = f"{input_len}_{bs}"
        latency_breakdown = sim_prefill(llm, hw_name, device_count, input_len, bs, compile_mode)
        throughput = bs * input_len / latency_breakdown.total
        f = open('prefill_results.csv', mode='a+')
        f.write(f"{input_len}, {bs}, {latency_breakdown.total}, {throughput}, {latency_breakdown.qkv_proj}, {latency_breakdown.q_mul_k}, {latency_breakdown.a_mul_v}, {latency_breakdown.o_matmul}, {latency_breakdown.ffn_matmul1}, {latency_breakdown.ffn_matmul2}, {latency_breakdown.ffn_matmul3}, {latency_breakdown.hadamard_mul}, {latency_breakdown.softmax}, {latency_breakdown.layernorm}, {latency_breakdown.activation}, {latency_breakdown.allreduce}\n")
        f.close()