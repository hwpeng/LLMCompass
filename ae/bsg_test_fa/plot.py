import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

categories = [
    "bs",
    "s",
    "latency",
]
categories += [
    "Q_K_V",
    "Q_mul_K",
    "A_mul_V",
    "Wo_proj",
    "W1_proj",
    "W2_proj",
    "Swi_proj",
    "Swi_mul",
    "Softmax",
    "LayerNorm_MHA",
    "LayerNorm_FFN",
    "GeLU",
    "AllReduce_MHA",
    "AllReduce_FFN",
]

plot_categories = [
    "Q_mul_K",
    "Softmax",
    "A_mul_V",
]

h100_prefill_std_attn = pd.read_csv(f"h100_prefill_std_attn.csv", header=None, names=categories)
h100_prefill_flash_attn = pd.read_csv( f"h100_prefill_flash_attn.csv", header=None, names=categories)
our_prefill_std_attn = pd.read_csv(f"3d_prefill_std_attn.csv", header=None, names=categories)
our_prefill_flash_attn = pd.read_csv( f"3d_prefill_flash_attn.csv", header=None, names=categories)
our_prefill_xcel_attn = pd.read_csv(f"3d_prefill_xcel_attn.csv", header=None, names=categories)
h100_decode_std_attn = pd.read_csv(f"h100_decode_std_attn.csv", header=None, names=categories)
h100_decode_flash_attn = pd.read_csv( f"h100_decode_flash_attn.csv", header=None, names=categories)
our_decode_std_attn = pd.read_csv(f"3d_decode_std_attn.csv", header=None, names=categories)
our_decode_flash_attn = pd.read_csv( f"3d_decode_flash_attn.csv", header=None, names=categories)
our_decode_xcel_attn = pd.read_csv(f"3d_decode_xcel_attn.csv", header=None, names=categories)


def plot(dfs, filename):
    df_labels = ["H100 STD", "H100 FA", "3D STD", "3D FA", "3D XCEL"]
    colors_matmul = sns.color_palette("flare_r", 6)
    colors_normalization = sns.color_palette("summer", 3)
    colors_gelu = sns.color_palette("pink", 1)
    colors_allreduce = sns.color_palette("Blues_r", 2)
    colors = colors_matmul + colors_normalization + colors_gelu + colors_allreduce

    plt.figure(figsize=(5, 3))

    # Create the stacked bar graph
    for x, row_index in enumerate(df_labels):
        values = dfs[x].iloc[0].tolist()
        bottom = 0
        for i, (category, value) in enumerate(zip(categories[3:], values[3:])):
            if category not in plot_categories:
                continue
            if row_index == df_labels[0]:
                plt.bar(
                    x, value, bottom=bottom, color=colors[i], label=category, width=0.5
                )
            else:
                plt.bar(x, value, bottom=bottom, color=colors[i], width=0.5)
            bottom += value

    plt.ylabel("Latency (s)")
    plt.xlabel("Configurations")
    plt.xticks(list(range(len(df_labels))), df_labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5)
    )
    plt.tight_layout()
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.show()


plot([h100_prefill_std_attn, h100_prefill_flash_attn, our_prefill_std_attn, our_prefill_flash_attn, our_prefill_xcel_attn], "figure_prefill.pdf")
plot([h100_decode_std_attn, h100_decode_flash_attn, our_decode_std_attn, our_decode_flash_attn, our_decode_xcel_attn], "figure_decode.pdf")
