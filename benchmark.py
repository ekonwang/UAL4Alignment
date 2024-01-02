import os
from pathlib import Path
import time


def main(
    ckpt_path: Path = Path("./out/lora/lima"),
    max_iters: int = 30999
):
    assert ckpt_path.is_dir()

    ckpts = sorted(os.listdir(ckpt_path))
    ckpts = [c for c in ckpts if ("finetuned" not in c and int(c.split('-')[1]) <= max_iters)]
    ckpts = [ckpt_path / c for c in ckpts]

    # run bash command
    for ckpt in ckpts:
        print(ckpt)
        print()
        cmd = f'python benchmark/benchmark_mc.py --data_dir ARC --lora_path {ckpt} --shot_num 5'
        os.system(cmd)
        cmd = f'python benchmark/benchmark_mc.py --data_dir TruthfulQA --lora_path {ckpt}'
        os.system(cmd)
        cmd = f'python benchmark/benchmark_mc.py --data_dir MMLU --lora_path {ckpt}'
        os.system(cmd)

        time.sleep(3)


if __name__ == '__main__':
    from jsonargparse.cli import CLI

    CLI(main)

