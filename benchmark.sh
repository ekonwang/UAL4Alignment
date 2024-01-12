# python benchmark/openllm_mistral.py --data_dir ARC --best_of 4
# python benchmark/openllm_mistral.py --data_dir TruthfulQA --best_of 4
# python benchmark/openllm_mistral.py --data_dir MMLU --best_of 4
# python benchmark/openllm_mistral.py --data_dir HellaSwag --best_of 4

python benchmark/openllm_vicuna.py --data_dir ARC --best_of 4
python benchmark/openllm_vicuna.py --data_dir TruthfulQA --best_of 4
python benchmark/openllm_vicuna.py --data_dir MMLU --best_of 4
python benchmark/openllm_vicuna.py --data_dir HellaSwag --best_of 4

