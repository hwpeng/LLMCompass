from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.transformer import (
    TransformerBlockAutoRegressionTP,
    TransformerBlockInitComputationTP,
)
from software_model.utils import Tensor, data_type_dict

this_dir = "ae/bsg_test_fa"

our_system = template_to_system(read_architecture_template("configs/3D_DRAM.json"))
h100_system = template_to_system(read_architecture_template("configs/GH100.json"))

def run_test(
    system,
    bs,
    seq_len,
    prefill_not_decode,
    use_flash_attn,
    use_flash_attn_xcel,
    output_filename,
    huristics="heuristic-GPU",  # None == roofline
):
    d_model = 12288
    n_heads = 96
    device_count = 4
    data_type = data_type_dict["fp16"]


    # Prefill
    if prefill_not_decode:
        model = TransformerBlockInitComputationTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=device_count,
            data_type=data_type,
            use_flash_attn=use_flash_attn,
            use_flash_attn_xcel=use_flash_attn_xcel,
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
            use_flash_attn_xcel=use_flash_attn_xcel,
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


heuristics = None
bs = 1
seq_len = 128 * 1024

for prefill_not_decode in [True, False]:
    t = "prefill" if prefill_not_decode else "decode"
    run_test(
        system=our_system,
        bs=bs,
        seq_len=seq_len,
        prefill_not_decode=prefill_not_decode,
        use_flash_attn=False,
        use_flash_attn_xcel=False,
        output_filename=f"{this_dir}/3d_{t}_std_attn.csv",
        huristics=heuristics,
    )
    run_test(
        system=our_system,
        bs=bs,
        seq_len=seq_len,
        prefill_not_decode=prefill_not_decode,
        use_flash_attn=True,
        use_flash_attn_xcel=False,
        output_filename=f"{this_dir}/3d_{t}_flash_attn.csv",
        huristics=heuristics,
    )
    run_test(
        system=our_system,
        bs=bs,
        seq_len=seq_len,
        prefill_not_decode=prefill_not_decode,
        use_flash_attn=True,
        use_flash_attn_xcel=True,
        output_filename=f"{this_dir}/3d_{t}_xcel_attn.csv",
        huristics=heuristics,
    )

    run_test(
        system=h100_system,
        bs=bs,
        seq_len=seq_len,
        prefill_not_decode=prefill_not_decode,
        use_flash_attn=False,
        use_flash_attn_xcel=False,
        output_filename=f"{this_dir}/h100_{t}_std_attn.csv",
        huristics=heuristics,
    )
    run_test(
        system=h100_system,
        bs=bs,
        seq_len=seq_len,
        prefill_not_decode=prefill_not_decode,
        use_flash_attn=True,
        use_flash_attn_xcel=False,
        output_filename=f"{this_dir}/h100_{t}_flash_attn.csv",
        huristics=heuristics,
    )
