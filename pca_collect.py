import seaborn as sns
from pathlib import Path
import random
import json

import torch
from lit_llama import Tokenizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 

sns.set(style="white", color_codes=True)


# some functions
def stat(labels):
    stat = {}
    for label in labels:
        if label not in stat:
            stat[label] = 0
        stat[label] += 1
    return stat


def select_high_freq_pairs_from_count(label_count):
    save_pairs = []
    for i in range(len(label_count)):
        for j in range(i+1, len(label_count)):
            save_pairs.append((label_count[i][0], label_count[j][0]))
    # random.shuffle(save_pairs)
    return save_pairs


def pca_trans(pair, features, normalize=False):
    collected_f = [(l, f) for l, f in features if l in pair]
    labels_t, features_t = zip(*collected_f)
    labels_t = list(labels_t)

    features_t = torch.stack(features_t).to(torch.float).numpy()
    pca = PCA(n_components=2)
    pca.fit(features_t)
    features_t = pca.transform(features_t)
    if normalize:
        scaler = StandardScaler()
        features_t = scaler.fit_transform(features_t)
    return features_t, labels_t


def silhouette_coef(labels, features):
    silhouette_avg = silhouette_score(features, labels)
    return silhouette_avg


def decode_pair(pair, tokenizer):
    return [
        tokenizer.processor.decode([pair[0]]), 
        tokenizer.processor.decode([pair[1]])
    ]


# some setup steps
def main(
    feat_path = "/cpfs01/user/yikunwang.p/workspace/lit-llama/out/pca_analysis/lima-7b/lima-test/lima-7b-ual-0.1-test-features.pt",
    index: int = 3,
):
    __tokenizer = "checkpoints/lit-llama/tokenizer.model"
    __features = feat_path

    tokenizer = Tokenizer(__tokenizer)
    features = torch.load(__features)
    labels = [f[0] for f in features]
    model_name = "-".join(__features.split('/')[-1].rsplit(".", 1)[0].split("-")[:-2])

    sns.displot(labels, bins=100, kde=True)
    plt.savefig('./out_dist.png', dpi=400)

    # sort the dict by value, descending
    labels_count = stat(labels)
    labels_count = sorted(labels_count.items(), key=lambda x: x[1], reverse=True)[:42]
    decoded_labels = {f"{tokenizer.processor.decode([l[0]])}": (f'#{l[0]}', l[1]) for l in labels_count}
    print('token | token_id | count')
    for d in decoded_labels:
        print(f"\t{d}: {decoded_labels[d]}")

    draw = False
    if draw:
        save_pairs = [
            (278, 304),
            (306, 366),
            (338, 526)
        ]
        chosen_pair = [
            373,
            29900
        ]
        decoded_pair = decode_pair(chosen_pair, tokenizer)
        print(f"chosen pair: {decoded_pair[0]} , {decoded_pair[1]}")

        feat, labels_t = pca_trans(chosen_pair, features)
        fig_path = __features.replace('.pt', '.pdf')

        # replace all pair[0] with 'Token A' and pair[1] with 'Token B'
        labels_t = [f'Token A' if l == chosen_pair[0] else f'Token B' for l in labels_t]
        plt.clf()
        # different shape for different tokens
        # fig = sns.scatterplot(x=feat[:,0], y=feat[:,1], hue=labels_t, legend='full')
        fig = sns.scatterplot(x=feat[:,0], y=feat[:,1], hue=labels_t, legend='full', style=labels_t)
        sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=True, offset=None, trim=False)
        # fig.set_xlim([-2, 2])
        # fig.set_ylim([-2, 2])
        # add grid on the x-axis and the y-axis
        fig.grid(True)
        # make the y ticks invisible while keep the y axis grid
        # if index > 0:
        #     fig.set_yticklabels([], visible=False)
        # remove the legend
        if index < 3:
            plt.legend([],[], frameon=False)
        plt.savefig(fig_path, dpi=400)
    else:
        normalize = False
        save_pairs = select_high_freq_pairs_from_count(labels_count)
        if not normalize:
            export_file = __features.replace('.pt', '.json')
        else:
            export_file = __features.replace('.pt', '.norm.json')
        export_list = []
        for i, pair in enumerate(tqdm(save_pairs)):
            decoded_pair = list(decode_pair(pair, tokenizer))
            feat, labels_t = pca_trans(pair, features, normalize=normalize)
            silhouette = silhouette_coef(labels_t, feat)
            export_list.append({
                'token_ids': list(pair),
                'tokens': decoded_pair,
                'silhouette coefficient': float(silhouette),
            })
            with open(export_file, 'w') as f:
                json.dump(export_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
