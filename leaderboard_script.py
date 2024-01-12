import os
from pathlib import Path
import time


def main(
    ckpt: Path = Path("./out/lora/lima"),
    max_iters: int = 6999,
    shot_configs: str = "0",
    best_of: int = 4,
    ckpt_policy: str = "steps",
    target_script: str = "benchmark/openllm_leaderboard.py",
):
    assert ckpt.is_dir()
    assert ckpt_policy in ["steps", "finetuned"]
    assert best_of > 0

    # larger epoch priority
    ckpts = reversed(sorted(os.listdir(ckpt)))
    if ckpt_policy == "steps":
        ckpts = [c for c in ckpts if (c.endswith(".pth") and "finetuned" not in c and int(c.split('-')[1]) <= max_iters)]
    else:
        ckpts = [c for c in ckpts if "finetuned" in c]
    ckpts = [ckpt / c for c in ckpts]

    shot_setting = shot_configs.split(',')
    shot_setting = [int(s) for s in shot_setting]

    dataset_configs = {
        'ARC': True, 
        'TruthfulQA': True, 
        'MMLU': True, 
        'HellaSwag': True
    }

    for shot_num in shot_setting:
        for ckpt in ckpts:
            for dataset_name, _valid in dataset_configs.items():
                if not _valid:
                    continue

                cmd = f'python {target_script} --data_dir {dataset_name} --lora_path {ckpt} --shot_num {shot_num} --best_of {best_of}'
                run_command(cmd)
                time.sleep(3)


# run bash command
    def run_command(cmd):
        ret = -1 
        while(ret):
            ret = os.system(cmd)
            print(cmd)
            if ret:
                time.sleep(60)


if __name__ == '__main__':
    from jsonargparse.cli import CLI

    CLI(main)

