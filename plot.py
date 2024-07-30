#  %%
import matplotlib.pyplot as plt
import numpy as np
from cc_utils import Latency
# %%
# Plot Prefill and Decode Latency and Throughput
decode_latency = dict()
decode_throughput = dict()
decode_breakdown = dict()
output_lens = []
batch_sizes = []
with open('decode_results_tmp.csv') as f:
    for line in f:
        if 'input_len' in line:
            continue
        input_len, output_len, bs, total, throughput, qkv_proj, q_mul_k, a_mul_v, o_matmul, ffn_matmul1, ffn_matmul2, ffn_matmul3, hadamard_mul, softmax, layernorm, activation, allreduce = line.strip().split(', ')
        if bs not in batch_sizes:
            batch_sizes.append(bs)
        if output_len not in output_lens:
            output_lens.append(output_len)
        key = f"{output_len}_{bs}"
        decode_latency[key] = float(total)
        decode_throughput[key] = float(throughput)
        decode_breakdown[key] = Latency(float(qkv_proj), float(q_mul_k), float(a_mul_v), float(o_matmul), float(ffn_matmul1), float(ffn_matmul2), float(ffn_matmul3), float(hadamard_mul), float(softmax), float(layernorm), float(activation), float(allreduce), float(total))

prefill_latency = dict()
prefill_throughput = dict()
prefill_breakdown = dict()
input_lens = []
with open('prefill_results_tmp.csv') as f:
    for line in f:
        if 'input_len' in line:
            continue
        input_len, bs, total, throughput, qkv_proj, q_mul_k, a_mul_v, o_matmul, ffn_matmul1, ffn_matmul2, ffn_matmul3, hadamard_mul, softmax, layernorm, activation, allreduce = line.strip().split(', ')
        if input_len not in input_lens:
            input_lens.append(input_len)
        key = f"{input_len}_{bs}"
        prefill_latency[key] = float(total)
        prefill_throughput[key] = float(throughput)
        prefill_breakdown[key] = Latency(float(qkv_proj), float(q_mul_k), float(a_mul_v), float(o_matmul), float(ffn_matmul1), float(ffn_matmul2), float(ffn_matmul3), float(hadamard_mul), float(softmax), float(layernorm), float(activation), float(allreduce), float(total))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# set spaces of subplots
plt.subplots_adjust(wspace=0.25, hspace=0.5)
for ax, phase in zip(axes, ['Prefill', 'Decode']):
    ax.set_title(f'{phase}')
    if phase == 'Prefill':
        latency = prefill_latency
        throughput = prefill_throughput
        breakdown = prefill_breakdown
        x = []
        x_labels = []
        offset = 0
        ax_r = ax.twinx()
        lat_norm = None
        tput_norm = None
        for input_len in input_lens:
            latency = []
            tput = []
            x_sub = []
            for bs in batch_sizes:
                x_labels.append(f"{int(input_len)//1024}K, {bs}")
                if lat_norm is None:
                    lat_norm = prefill_latency[f"{input_len}_{bs}"]
                    tput_norm = prefill_throughput[f"{input_len}_{bs}"]
                latency.append(prefill_latency[f"{input_len}_{bs}"] / lat_norm)
                tput.append(prefill_throughput[f"{input_len}_{bs}"] / tput_norm)
                x.append(offset)
                x_sub.append(offset)
                offset += 1
            ax.plot(x_sub, tput, marker='x', color='blue')
            ax_r.plot(x_sub, latency, marker='o', color='red')
        ax.set_xlabel('Input Length and Batch Size')
        ax_r.set_ylabel('Latency')
        l = ax.plot(x_sub, tput, marker='x', color='blue', label='Throughput')
        r = ax_r.plot(x_sub, latency, marker='o', color='red', label='Latency')
        labs = [l.get_label() for l in l+r]
        ax.legend(l+r, labs, loc='upper center')
    else:
        latency = decode_latency
        throughput = decode_throughput
        breakdown = decode_breakdown
        x = []
        x_labels = []
        offset = 0
        tput_norm = None
        for output_len in output_lens:
            tput = []
            x_sub = []
            for bs in batch_sizes:
                x_labels.append(f"{int(output_len)//1024}K, {bs}")
                x.append(f"{output_len}_{bs}")
                if tput_norm is None:
                    tput_norm = decode_throughput[f"{output_len}_{bs}"]
                tput.append(decode_throughput[f"{output_len}_{bs}"] / tput_norm)
                x.append(offset)
                x_sub.append(offset)
                offset += 1
            ax.plot(x_sub, tput, marker='x', color='blue', label='Throughput')
        ax.set_xlabel('Output Length and Batch Size')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_ylabel('Throughput')
    # ax.legend()
    

# %%
