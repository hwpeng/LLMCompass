from cc_utils import get_model_params
import argparse
from cc_utils import sim_prefill_fast, sim_decode_fast

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefill test")
    parser.add_argument("--llm", type=str, default="llama3-70B", help="LLM model")
    parser.add_argument("--phase", type=str, default="prefill", help="Prefill or decode")
    parser.add_argument("--hw_name", type=str, default="GH100", help="Hardware name")
    parser.add_argument("--device_count", type=int, default=8, help="Number of devices")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--max_input_len", type=int, default=128, help="Maximum input length in K tokens")
    parser.add_argument("--max_output_len", type=int, default=128, help="Maximum output length in K tokens")
    parser.add_argument("--flash_attn", action="store_true", help="Use flash attention")
    parser.add_argument("--output_file", type=str, default=None, help="Output file")

    args = parser.parse_args()
    llm = args.llm
    phase = args.phase
    d_model, d_ffn, n_heads, n_kv_heads, activation, n_layers = get_model_params(llm)
    hw_name = args.hw_name
    device_count = args.device_count

    if args.output_file is None:
        if args.flash_attn:
            output_file = f"{phase}_{llm}-FA_{device_count}_{hw_name}.csv"
        else:
            output_file = f"{phase}_{llm}_{device_count}_{hw_name}.csv"
    else:
        output_file = args.output_file
    

    batch_sizes = []
    bs = 1
    while bs <= args.max_batch_size:
        batch_sizes.append(bs)
        bs *= 2

    input_lens = []
    input_len = min(args.max_input_len, 4)
    while input_len <= args.max_input_len:
        input_lens.append(input_len * 1024)
        input_len *= 2
    
    output_lens = []
    output_len = min(args.max_output_len, 4)
    while output_len <= args.max_output_len:
        output_lens.append(output_len * 1024)
        output_len *= 2

    if phase == "prefill":
        latency_breakdown_list = sim_prefill_fast(llm, hw_name, device_count, input_lens, batch_sizes, args.flash_attn)
        with open(output_file, mode='w') as f:
            f.write("input_len, bs, total, throughput, qkv_proj, q_mul_k, a_mul_v, o_matmul, ffn_matmul1, ffn_matmul2, ffn_matmul3, hadamard_mul, softmax, layernorm, activation, allreduce\n")
            for bs in batch_sizes:
                for input_len in input_lens:
                    key = f"{input_len}_{bs}"
                    latency_breakdown = latency_breakdown_list[key]
                    throughput = bs * input_len / latency_breakdown.total
                    f.write(f"{input_len}, {bs}, {latency_breakdown.total}, {throughput}, {latency_breakdown.qkv_proj}, {latency_breakdown.q_mul_k}, {latency_breakdown.a_mul_v}, {latency_breakdown.o_matmul}, {latency_breakdown.ffn_matmul1}, {latency_breakdown.ffn_matmul2}, {latency_breakdown.ffn_matmul3}, {latency_breakdown.hadamard_mul}, {latency_breakdown.softmax}, {latency_breakdown.layernorm}, {latency_breakdown.activation}, {latency_breakdown.allreduce}\n")
    elif phase == "decode":
        latency_breakdown_list = sim_decode_fast(llm, hw_name, device_count, input_lens, output_lens, batch_sizes, args.flash_attn)
        with open(output_file, mode='w') as f:
            f.write("input_len, output_len, bs, total, throughput, qkv_proj, q_mul_k, a_mul_v, o_matmul, ffn_matmul1, ffn_matmul2, ffn_matmul3, hadamard_mul, softmax, layernorm, activation, allreduce\n")
            for bs in batch_sizes:
                for input_len in input_lens:
                    for output_len in output_lens:
                        key = f"{input_len}_{output_len}_{bs}"
                        latency_breakdown = latency_breakdown_list[key]
                        throughput = bs * output_len / latency_breakdown.total
                        f.write(f"{input_len}, {output_len}, {bs}, {latency_breakdown.total}, {throughput}, {latency_breakdown.qkv_proj}, {latency_breakdown.q_mul_k}, {latency_breakdown.a_mul_v}, {latency_breakdown.o_matmul}, {latency_breakdown.ffn_matmul1}, {latency_breakdown.ffn_matmul2}, {latency_breakdown.ffn_matmul3}, {latency_breakdown.hadamard_mul}, {latency_breakdown.softmax}, {latency_breakdown.layernorm}, {latency_breakdown.activation}, {latency_breakdown.allreduce}\n")
    else:
        print("Invalid phase")
