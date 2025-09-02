import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV_FILE = "grammaticality_results.csv"
OUTPUT_PLOT_FILE = "final_tsm_ranking_plot.png"

PROBE_MAPPING = {
    'arbitrary_third_person': '3rd Person (-ik)',
    'arbitrary_past_tense': 'Past Tense (-ul)',
    'arbitrary_plural': 'Plurality (-im)',
    'arbitrary_possessive': 'Possessive (-eph)',
    'arbitrary_progressive': 'Progressive (-ov)',
    'infinitive': 'Basic Syntax',
    'adjective': 'Basic Syntax',
    'verb_base_form': 'Basic Syntax'
}

MASTERY_THRESHOLD = 0.90
STABILITY_WINDOW = 500

def calculate_tsm(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个语法类别的“稳定掌握时间”(Time to Stable Mastery)"""
    print("正在计算稳定掌握时间 (TSM)...")
    
    tsm_scores = {}
    for category, group in df.groupby('category'):
        avg_accuracy = group.groupby('step')['accuracy'].mean()
        
        is_mastered = avg_accuracy >= MASTERY_THRESHOLD
        
        step_interval = avg_accuracy.index[1] - avg_accuracy.index[0] if len(avg_accuracy.index) > 1 else 1
        window_size = max(1, STABILITY_WINDOW // step_interval)
        is_stable = is_mastered.rolling(window=window_size, min_periods=window_size).sum() >= window_size

        stable_points = avg_accuracy.index[is_stable]
        
        if not stable_points.empty:
            tsm_scores[category] = stable_points[0]
        else:
            tsm_scores[category] = df['step'].max() * 1.1 

    tsm_df = pd.Series(tsm_scores).reset_index()
    tsm_df.columns = ['category', 'TSM_Score']
    
    tsm_df = tsm_df.sort_values(by='TSM_Score', ascending=True)
    
    print("\nTSM 排名计算完成：")
    print(tsm_df)
    
    return tsm_df

def plot_final_ranking(df: pd.DataFrame):
    """绘制最终的、基于TSM的排序条形图"""
    print("正在生成最终排名图表...")
    
    plt.rc('font', **{'family': 'Arial', 'size': 12, 'weight': 'normal'})
    plt.rc('axes', labelcolor='black', titlecolor='black')
    plt.rc('xtick', color='black')
    plt.rc('ytick', color='black')

    sns.set_theme(style="whitegrid")
    
    FIXED_WIDTH = 12.0

    num_categories = len(df)
    height_per_category = 0.6
    base_height = 2.0
    dynamic_height = (num_categories * height_per_category) + base_height
    
    fig, ax = plt.subplots(figsize=(FIXED_WIDTH, dynamic_height))

    palette = sns.color_palette("coolwarm", n_colors=len(df))
    
    sns.barplot(
        x='TSM_Score',
        y='category',
        data=df,
        palette=palette,
        orient='h',
        order=df['category'],
        ax=ax
    )

    ax.set_title(
        "Final Acquisition Order Based on Time to Stable Mastery (TSM)",
        fontfamily='Arial', fontsize=18, fontweight='bold', color='black', pad=20
    )
    
    ax.set_xlabel(
        "Training Steps Required to Reach Stable Mastery (Fewer is Easier)",
        fontfamily='Arial', fontsize=16, fontweight='normal', color='black', labelpad=15
    )
    ax.set_ylabel(
        "Grammatical Category",
        fontfamily='Arial', fontsize=16, fontweight='normal', color='black', labelpad=15
    )
    
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)
    
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('Arial')
        label.set_color('black')

    for label in ax.get_xticklabels():
        label.set_fontweight('normal')
        label.set_fontfamily('Arial')
        label.set_color('black')

    fig.subplots_adjust(left=0.3)

    fig.savefig(OUTPUT_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"最终 TSM 排名图表已保存至：{OUTPUT_PLOT_FILE}")


if __name__ == "__main__":
    try:
        df = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        print(f"错误：未找到数据文件 '{INPUT_CSV_FILE}'。")
        print("正在创建用于测试的模拟数据...")
        data = {
            'step': list(range(0, 10000, 100)) * 6,
            'probe_name': (['arbitrary_plural'] * 100 + 
                           ['infinitive'] * 100 + 
                           ['arbitrary_past_tense'] * 100 +
                           ['adjective'] * 100 +
                           ['arbitrary_progressive'] * 100 +
                           ['arbitrary_third_person'] * 100),
            'accuracy': ([min(1, x/2000) for x in range(0, 10000, 100)] +
                         [min(1, x/3000) for x in range(0, 10000, 100)] +
                         [min(1, x/5000) for x in range(0, 10000, 100)] +
                         [min(1, x/4000) for x in range(0, 10000, 100)] +
                         [min(1, x/8000) for x in range(0, 10000, 100)] +
                         [min(1, x/9500) for x in range(0, 10000, 100)])
        }
        df = pd.DataFrame(data)
        df.to_csv(INPUT_CSV_FILE, index=False)
        print(f"模拟数据已保存至 '{INPUT_CSV_FILE}'。")

    df['category'] = df['probe_name'].map(PROBE_MAPPING)
    df.dropna(subset=['category'], inplace=True)

    tsm_ranking_df = calculate_tsm(df.copy())
    
    plot_final_ranking(tsm_ranking_df)