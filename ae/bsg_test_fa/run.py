from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.transformer import (
    TransformerBlockAutoRegressionTP,
    TransformerBlockInitComputationTP,
)
from software_model.utils import Tensor, data_type_dict

this_dir = "ae/bsg_test_fa"

A100_specs = read_architecture_template("configs/GA100.json")
A100_system = template_to_system(A100_specs)

print(f"{A100_system.device.compute_module.total_vector_flops=}")
print(f"{A100_system.device.compute_module.total_systolic_array_flops=}")


def run_test(
    name,
    system,
    bs,
    seq_len,
    prefill_not_decode,
    use_flash_attn,
    huristics="heuristic-GPU",  # None == roofline
):
    d_model = 12288
    n_heads = 96
    device_count = 1
    data_type = data_type_dict["fp16"]

    output_filename = f"{this_dir}/{name}_{'prefill' if prefill_not_decode else 'decode'}_{'flashattn' if use_flash_attn else 'stdattn'}_{'roofline' if huristics is None else 'simulated'}.csv"

    # Prefill
    if prefill_not_decode:
        model = TransformerBlockInitComputationTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=device_count,
            data_type=data_type,
            use_flash_attn=use_flash_attn,
        )
        _ = model(Tensor([bs, seq_len, d_model], data_type))

    # Decode
    else:
        model = TransformerBlockAutoRegressionTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=device_count,
            data_type=data_type,
            use_flash_attn=use_flash_attn,
        )
        _ = model(Tensor([bs, 1, d_model], data_type), seq_len)

    # simulated
    if huristics is not None:
        simulated = model.compile_and_simulate(system, heuristics)
        with open(output_filename, "w") as f:
            f.write(f"{bs}, {seq_len}, {simulated}, {model.simluate_log}\n")

    # roofline
    else:
        simulated = model.roofline_model(system)
        with open(output_filename, "w") as f:
            f.write(
                # f"{bs}, {seq_len}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
                f"{bs}, {seq_len}, {simulated}, {model.roofline_log}\n"
            )


name = f"A100"
heuristics = None
bs = 1
seq_len = 128 * 1024

run_test(
    name=name,
    system=A100_system,
    bs=bs,
    seq_len=seq_len,
    prefill_not_decode=True,
    use_flash_attn=False,
    huristics=heuristics,
)
run_test(
    name=name,
    system=A100_system,
    bs=bs,
    seq_len=seq_len,
    prefill_not_decode=True,
    use_flash_attn=True,
    huristics=heuristics,
)
