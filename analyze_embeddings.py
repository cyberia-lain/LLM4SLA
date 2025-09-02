import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from adjustText import adjust_text
from matplotlib.lines import Line2D

CHECKPOINTS_DIR = "./wug_test_results"
OUTPUT_TRAJECTORY_PLOT = "Embedding_Galaxy_Plot.png"
OUTPUT_DISTANCE_PLOT = "Category_Distance_Community.png"

TARGET_WORDS = {
    "New Verb": ["prallut", "vrigorn"],
    "New Noun": ["broukth", "rilthex"],
    "Control Word": ["Kthwo"],
    "Reference Verbs": [
        "run", "throw", "build", "be", "seem",
        "know", "think", "believe", "see", "hear"
    ],
    "Reference Nouns": [
        "cat", "table", "water", "car", "hand",
        "idea", "time", "love", "reason", "system"
    ]
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
})

def get_embeddings_df():
    checkpoint_paths = [os.path.join(CHECKPOINTS_DIR, d) for d in os.listdir(CHECKPOINTS_DIR) if d.startswith('checkpoint')]
    if not checkpoint_paths:
        print(f"错误：在目录 '{CHECKPOINTS_DIR}' 中未找到任何检查点。")
        return pd.DataFrame()
    checkpoint_paths.sort(key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))

    all_embeddings = []
    for path in tqdm(checkpoint_paths, desc="正在提取词嵌入"):
        step = int(re.search(r'checkpoint-(\d+)', path).group(1))
        model = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)

        for category, words in TARGET_WORDS.items():
            token_ids = tokenizer(words, add_special_tokens=False, padding=True, return_tensors="pt")['input_ids']
            with torch.no_grad():
                embeddings = model.embeddings.word_embeddings(token_ids).mean(dim=1).cpu().numpy()

            for i, word in enumerate(words):
                all_embeddings.append({
                    "step": step, "word": word, "category": category, "embedding": embeddings[i]
                })
    return pd.DataFrame(all_embeddings)

def plot_trajectory_galaxy(df: pd.DataFrame):
    print("正在生成词嵌入轨迹图...")

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
    embedding_matrix = np.vstack(df['embedding'].values)
    df[['x', 'y']] = tsne.fit_transform(embedding_matrix)

    final_step = df['step'].max()
    df_final = df[df['step'] == final_step]

    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    ax = plt.gca()

    ref_verbs_final = df_final[df_final['category'] == 'Reference Verbs']
    ax.scatter(ref_verbs_final['x'], ref_verbs_final['y'], s=1000, marker='o', color='lightblue', alpha=0.8, edgecolors='black', linewidth=0.5)

    ref_nouns_final = df_final[df_final['category'] == 'Reference Nouns']
    ax.scatter(ref_nouns_final['x'], ref_nouns_final['y'], s=1000, marker='s', color='lightcoral', alpha=0.8, edgecolors='black', linewidth=0.5)

    texts_ref = []
    for _, row in pd.concat([ref_verbs_final, ref_nouns_final]).iterrows():
        texts_ref.append(ax.text(row['x'], row['y'], row['word'], fontsize=15, fontweight='bold', ha='center', va='center'))
    adjust_text(texts_ref, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))

    trajectory_colors = {'New Verb': 'blue', 'New Noun': 'red', 'Control Word': 'black'}
    for word in TARGET_WORDS['New Verb'] + TARGET_WORDS['New Noun'] + TARGET_WORDS['Control Word']:
        word_df = df[df['word'] == word].sort_values('step')
        if word_df.empty: continue

        category = word_df['category'].iloc[0]
        color = trajectory_colors[category]

        points = word_df[['x', 'y']].values
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=4, alpha=0.8, zorder=3)
        ax.scatter(points[0, 0], points[0, 1], color=color, s=400, marker='o', edgecolor='white', linewidth=1.5, zorder=4)
        ax.scatter(points[-1, 0], points[-1, 1], color=color, s=800, marker='X', edgecolor='white', linewidth=1.5, zorder=5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Reference Verbs', markerfacecolor='lightblue', markersize=22, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='Reference Nouns', markerfacecolor='lightcoral', markersize=22, markeredgecolor='black'),
        Line2D([0], [0], color='white', label='', markersize=0),
        Line2D([0], [0], color='blue', lw=4, label='New Verb Trajectory'),
        Line2D([0], [0], color='red', lw=4, label='New Noun Trajectory'),
        Line2D([0], [0], color='black', lw=4, label='Control Word Trajectory'),
        Line2D([0], [0], color='white', label='', markersize=0),
        Line2D([0], [0], marker='o', color='w', label='Trajectory Start', markerfacecolor='gray', markeredgecolor='white', markersize=18),
        Line2D([0], [0], marker='X', color='w', label='Trajectory End', markerfacecolor='gray', markeredgecolor='white', markersize=20),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=14,
        labelspacing=1.5,
        borderaxespad=0.
    )
    legend.get_frame().set_alpha(0.9)

    ax.set_title("Grammatical Category Induction in Semantic Space", fontsize=22, pad=20, weight='bold')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=18, weight='bold')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=18, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    plt.savefig(OUTPUT_TRAJECTORY_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"词嵌入图已保存至: {OUTPUT_TRAJECTORY_PLOT}")


def plot_distance_convergence(df: pd.DataFrame):
    print("正在生成图...")

    ref_verb_centroid_vec = df[df['category'] == 'Reference Verbs']['embedding'].apply(pd.Series).mean().values.reshape(1, -1)
    ref_noun_centroid_vec = df[df['category'] == 'Reference Nouns']['embedding'].apply(pd.Series).mean().values.reshape(1, -1)
    distances = []
    for step, group in df.groupby('step'):
        for word_type in ['New Verb', 'New Noun']:
            if not group[group['category'] == word_type].empty:
                vecs = np.vstack(group[group['category'] == word_type]['embedding'].values)
                dist_to_verb = 1 - cosine_similarity(vecs, ref_verb_centroid_vec).mean()
                dist_to_noun = 1 - cosine_similarity(vecs, ref_noun_centroid_vec).mean()
                distances.append({'step': step, 'type': word_type, 'dist_to_verb': dist_to_verb, 'dist_to_noun': dist_to_noun})
    dist_df = pd.DataFrame(distances)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    sns.set_theme(style="whitegrid")

    verb_dist_data = dist_df[dist_df['type'] == 'New Verb']
    axes[0].plot(verb_dist_data['step'], verb_dist_data['dist_to_verb'], color='blue', linewidth=3.5, label='Distance to Verb Community')
    axes[0].plot(verb_dist_data['step'], verb_dist_data['dist_to_noun'], color='red', linestyle='--', linewidth=3.5, label='Distance to Noun Community')
    axes[0].set_title("New Verbs Converge Towards Verb Community", fontsize=18, pad=15, weight='bold')
    axes[0].set_ylabel("Avg. Cosine Distance", fontsize=16, weight='bold')
    axes[0].legend(fontsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    noun_dist_data = dist_df[dist_df['type'] == 'New Noun']
    axes[1].plot(noun_dist_data['step'], noun_dist_data['dist_to_noun'], color='red', linewidth=3.5, label='Distance to Noun Community')
    axes[1].plot(noun_dist_data['step'], noun_dist_data['dist_to_verb'], color='blue', linestyle='--', linewidth=3.5, label='Distance to Verb Community')
    axes[1].set_title("New Nouns Converge Towards Noun Community", fontsize=18, pad=15, weight='bold')
    axes[1].set_xlabel("Training Steps", fontsize=16, weight='bold', labelpad=15)
    axes[1].set_ylabel("Avg. Cosine Distance", fontsize=16, weight='bold')
    axes[1].legend(fontsize=14)
    axes[1].tick_params(axis='both', labelsize=14)
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DISTANCE_PLOT, dpi=300)
    plt.close()
    print(f"已保存至: {OUTPUT_DISTANCE_PLOT}")

if __name__ == "__main__":
    try:
        import adjustText
    except ImportError:
        print("正在安装依赖库 adjustText...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "adjustText"])
        print("adjustText 安装成功。")

    embeddings_df = get_embeddings_df()
    if not embeddings_df.empty:
        plot_trajectory_galaxy(embeddings_df.copy())
        plot_distance_convergence(embeddings_df.copy())