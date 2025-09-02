 code Pythondownloadcontent_copyexpand_less    import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

INPUT_CSV_FILE = "grammaticality_results.csv"
OUTPUT_PLOT_FILE = "heatmap.png"

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13

PROBE_MAPPING = {
    'arbitrary_third_person': 'Agreement (-ik)',
    'long_distance_agreement': 'Agreement (-ik)',
    'plural_agreement': 'Agreement (-ik)',
    'arbitrary_past_tense': 'Tense (-ul)',
    'arbitrary_plural': 'Plurality (-im)',
    'arbitrary_possessive': 'Possessive (-eph)',
    'arbitrary_progressive': 'Progressive (-ov)',
    'infinitive': 'Basic Syntax',
    'adjective': 'Basic Syntax',
    'verb_base_form': 'Basic Syntax'
}

KRASHEN_GROUPS_ORDER = [
    'Progressive (-ov)',
    'Plurality (-im)',
    'Basic Syntax',
    'Tense (-ul)',
    'Agreement (-ik)',
    'Possessive (-eph)'
]

SMOOTHING_WINDOW = 7

def plot_convergence_heatmap(df: pd.DataFrame):
    """
    绘制收敛区域图，以类比克拉申的习得顺序分组。
    """
    print("正在生成收敛区域图...")

    df['category'] = df['probe_name'].map(PROBE_MAPPING)
    
    if df['category'].isnull().any():
        print("警告：发现未定义的 probe_name，将被忽略：")
        print(df[df['category'].isnull()]['probe_name'].unique())
        df.dropna(subset=['category'], inplace=True)

    df['accuracy_smoothed'] = df.groupby('category')['accuracy'].transform(
        lambda x: x.rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean()
    )
    
    pivot_df = df.pivot_table(
        index='category', 
        columns='step', 
        values='accuracy_smoothed'
    )
    
    ordered_categories = [cat for cat in KRASHEN_GROUPS_ORDER if cat in pivot_df.index]
    pivot_df = pivot_df.reindex(ordered_categories)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 9)) 

    sns.heatmap(
        pivot_df,
        ax=ax,
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': 'Grammaticality Judgment Accuracy'}
    )

    ax.set_title(
        "Heatmap of Grammatical Acquisition Order", 
        fontsize=18, 
        pad=20, 
        weight='bold'
    )
    
    ax.set_xlabel("Training Steps", fontsize=14, labelpad=15)
    ax.set_ylabel("Grammatical Category (Grouped by SLA Theory)", fontsize=14, labelpad=15, weight='bold')
    
    ax.tick_params(axis='y', labelrotation=0)
    ax.tick_params(axis='x', rotation=90)

    num_group1 = len([cat for cat in KRASHEN_GROUPS_ORDER[:3] if cat in ordered_categories])
    ax.axhline(y=num_group1, color='black', linestyle='--', linewidth=3)
    
    ax.text(pivot_df.columns.max() * 1.01, num_group1 / 2, 'Group 1:\nEarly Acquired\n(e.g., -ing, Plural)', 
            ha='left', va='center', fontsize=16, fontweight='bold')
    
    ax.text(pivot_df.columns.max() * 1.01, num_group1 + (len(ordered_categories) - num_group1) / 2, 'Group 4:\nLate Acquired\n(e.g., -ed, 3rd-s, Possessive)', 
            ha='left', va='center', fontsize=16, fontweight='bold')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.yaxis.label.set_size(14)

    plt.tight_layout(rect=[0, 0.05, 0.85, 1])
    
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
    plt.close()
    print(f"图像已保存至: {OUTPUT_PLOT_FILE}")


if __name__ == "__main__":
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        print(f"错误：未找到数据文件 '{INPUT_CSV_FILE}'。")
        exit()
    
    plot_convergence_heatmap(df.copy())
  