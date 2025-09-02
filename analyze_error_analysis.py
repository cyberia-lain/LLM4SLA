import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

CHECKPOINTS_DIR = "./wug_test_results"
OUTPUT_CSV_FILE = "TopK_Predictions.csv"
OUTPUT_PLOT_FILE = "PTopK_Evolution_Plot.png"

PROBE_SENTENCE = "The scientist wants to [MASK]."
CANDIDATE_WORDS = [
    "prallut",
    "vrigorn",
    "run",
    "the",
    "Kthwo"
]
TOP_N_TO_CHECK = 20
SMOOTHING_WINDOW = 11


def smooth_series(series, window=11):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def set_plot_style():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['legend.title_fontsize'] = 16


def analyze_and_visualize_topk():
    print("开始分析...")

    checkpoint_paths = [os.path.join(CHECKPOINTS_DIR, d) for d in os.listdir(CHECKPOINTS_DIR) if d.startswith('checkpoint')]
    if not checkpoint_paths:
        print(f"错误：在目录 '{CHECKPOINTS_DIR}' 中没有找到任何检查点。")
        return
    checkpoint_paths.sort(key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))

    all_ranks = []

    for path in tqdm(checkpoint_paths, desc="正在分析检查点"):
        step = int(re.search(r'checkpoint-(\d+)', path).group(1))
        model = AutoModelForMaskedLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)

        inputs = tokenizer(PROBE_SENTENCE, return_tensors="pt")
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        with torch.no_grad():
            logits = model(**inputs).logits

        mask_token_logits = logits[0, mask_token_index, :]
        top_n_indices = torch.topk(mask_token_logits, TOP_N_TO_CHECK, dim=1).indices[0].tolist()

        rank_record = {"step": step}
        for word in CANDIDATE_WORDS:
            try:
                token_id = tokenizer.convert_tokens_to_ids(word)
                rank = top_n_indices.index(token_id) + 1
            except ValueError:
                rank = TOP_N_TO_CHECK + 1
            rank_record[word] = rank
        all_ranks.append(rank_record)

    df = pd.DataFrame(all_ranks)
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\n数据已经保存到: {OUTPUT_CSV_FILE}")

    print("正在生...")
    set_plot_style()

    df_long = df.melt(id_vars=['step'], value_vars=CANDIDATE_WORDS,
                      var_name='word', value_name='rank')

    df_long['rank_smoothed'] = df_long.groupby('word')['rank'].transform(smooth_series, window=SMOOTHING_WINDOW)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    lineplot = sns.lineplot(
        data=df_long,
        x='step',
        y='rank_smoothed',
        hue='word',
        palette='tab10',
        linewidth=2.5,
        style='word',
        markers=False,
        dashes=False,
    )

    plt.gca().invert_yaxis()
    plt.yticks(range(1, TOP_N_TO_CHECK + 2, 2))

    plt.title("Evolution of Top-K Predictions for '[MASK]'", pad=15)
    plt.xlabel("Training Steps", labelpad=12)
    plt.ylabel("Prediction Rank (Lower is Better)", labelpad=12, weight='bold')

    legend = plt.legend(title="Candidate Words")
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_alpha(1.0)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
    plt.close()
    print(f"已经保存到: {OUTPUT_PLOT_FILE}")


if __name__ == "__main__":
    analyze_and_visualize_topk()
  