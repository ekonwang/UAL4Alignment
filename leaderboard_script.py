import os
from pathlib import Path
import time


def main(
    ckpt: Path = Path("./out/lora/lima"),
    max_iters: int = 6999,
    shot_configs: str = "0",
    best_of: int = 4,
):
    assert ckpt.is_dir()

    # larger epoch priority
    ckpts = reversed(sorted(os.listdir(ckpt)))
    ckpts = [c for c in ckpts if ("finetuned" not in c and int(c.split('-')[1]) <= max_iters)]
    ckpts = [ckpt / c for c in ckpts]

    shot_setting = shot_configs.split(',')
    shot_setting = [int(s) for s in shot_setting]

    # run bash command
    def run_command(cmd):
        ret = -1 
        while(ret):
            ret = os.system(cmd)
            print(cmd)
            if ret:
                time.sleep(60)

    for shot_num in shot_setting:
        for ckpt in ckpts:
            print(ckpt)
            print()
            # cmd = f'python benchmark/openllm_leaderboard.py --data_dir ARC --lora_path {ckpt} --shot_num {shot_num} --best_of {best_of}'
            # run_command(cmd)
            # cmd = f'python benchmark/openllm_leaderboard.py --data_dir TruthfulQA --lora_path {ckpt} --shot_num {shot_num} --best_of {best_of}'
            # run_command(cmd)
            # cmd = f'python benchmark/openllm_leaderboard.py --data_dir MMLU --lora_path {ckpt} --shot_num {shot_num} --best_of {best_of}'
            # run_command(cmd)
            cmd = f'python benchmark/openllm_leaderboard.py --data_dir HellaSwag --lora_path {ckpt} --shot_num {shot_num} --best_of {best_of}'
            run_command(cmd)

            time.sleep(3)


if __name__ == '__main__':
    from jsonargparse.cli import CLI

    CLI(main)

