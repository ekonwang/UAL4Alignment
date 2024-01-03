LORA_ALPACA=/workspace/lit-llama/out/lora/alpaca/lit-llama-lora-finetuned.pth
LORA_LIMA_E15=/workspace/lit-llama/out/lora/lima/sft_lima_lora_sctx-512_multi-dialogue-micro1_epoch60_ls-0.10_23-12-26-07-15-07/iter-015359-ckpt.pth
LORA_LIMA_E20=/workspace/lit-llama/out/lora/lima/sft_lima_lora_sctx-512_multi-dialogue-micro1_epoch60_ls-0.10_23-12-26-07-15-07/iter-020479-ckpt.pth
LORA_ADA_LIMA_E20=/workspace/lit-llama/out/lora/lima/sft_ada-lima_lora_sctx-512_ls-0.10_23-12-28-08-32-26/iter-020479-ckpt.pth


# python benchmark/benchmark_mc.py --data_dir "ARC" --shot_num 5
# python benchmark/benchmark_mc.py --data_dir "TruthfulQA"
python benchmark/benchmark_mc.py --data_dir "MMLU" --shot_num 5

for lora_path in $LORA_ALPACA $LORA_LIMA_E15 $LORA_LIMA_E20 $LORA_ADA_LIMA_E20
do
    # python benchmark/benchmark_mc.py --data_dir "ARC" --lora_path $lora_path --shot_num 5
    # python benchmark/benchmark_mc.py --data_dir "TruthfulQA" --lora_path $lora_path
    python benchmark/benchmark_mc.py --data_dir "MMLU" --lora_path $lora_path --shot_num 5
done
