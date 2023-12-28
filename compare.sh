ALPACA_PATH=/workspace/lit-llama/out/lora/alpaca/alpaca-23-12-16-16-12-45.json
Epoch30=/workspace/lit-llama/out/lora/lima/lima-lima-lit-llama-lora-finetuned-23-12-26-11-52-50-30epoch.json
Epoch25=/workspace/lit-llama/out/lora/lima/lima-sft_lima_lora_sctx-512_micro1_23-12-25-17-35-16-iter-025599-ckpt-23-12-27-17-06-17.json
Epoch20=/workspace/lit-llama/out/lora/lima/lima-sft_lima_lora_sctx-512_micro1_23-12-25-17-35-16-iter-iter-020479-ckpt.json
Epoch15=/workspace/lit-llama/out/lora/lima/lima-iter-015359-ckpt-23-12-26-11-24-09.json
Epoch10=/workspace/lit-llama/out/lora/lima/lima-iter-010239-ckpt-23-12-26-11-22-42.json
Epoch15_Ada=/workspace/lit-llama/out/lora/lima/lima-ada-lima_lora_sctx-512_ls-0.10_23-12-26-13-04-52-lit-llama-lora-finetuned-23-12-27-09-32-04.json



python benchmark/gpt4_lima_compare_v2.py --max_turns 150 --first_path $Epoch30 --second_path $ALPACA_PATH --tag "30es-vs-alpaca"
python benchmark/gpt4_lima_compare_v2.py --max_turns 150 --first_path $Epoch15 --second_path $ALPACA_PATH --tag "15es-vs-alpaca"
python benchmark/gpt4_lima_compare_v2.py --max_turns 150 --first_path $Epoch10 --second_path $ALPACA_PATH --tag "10es-vs-alpaca"
python benchmark/gpt4_lima_compare_v2.py --max_turns 150 --first_path ${Epoch15_Ada} --second_path $ALPACA_PATH --tag "15es-ada-vs-alpaca"

python benchmark/gpt4_lima_compare_v2.py --max_turns 100 --first_path $Epoch30 --second_path ${Epoch15_Ada} --tag "30es-vs-15es-ada"

python benchmark/gpt4_lima_compare_v2.py --max_turns 100 --first_path $Epoch20 --second_path $ALPACA_PATH --tag "20es-vs-alpaca"
python benchmark/gpt4_lima_compare_v2.py --max_turns 100 --first_path $Epoch25 --second_path $ALPACA_PATH --tag "25es-vs-alpaca"

python benchmark/gpt4_lima_compare_v2.py --max_turns 100 --first_path $Epoch30 --second_path $Epoch25 --tag "30es-vs-25es"
python benchmark/gpt4_lima_compare_v2.py --max_turns 100 --first_path $Epoch30 --second_path $Epoch20 --tag "30es-vs-20es"

