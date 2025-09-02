好的，这是移除所有注释并简化了控制台中文输出后的脚本。脚本的其他部分未做任何修改。 code Pythondownloadcontent_copyexpand_less    import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV_FILE = "control_test_results.csv"
OUTPUT_PLOT_FILE = "control_word_suppression_accuracy.png"
PLOT_TITLE = "Model's Final Accuracy in Rejecting Ungrammatical Control Words"

def plot_final_accuracy(df: pd.DataFrame):
    if 'accuracy' not in df.columns:
        print(f"错误：在CSV文件中未找到 'accuracy' 列。")
        return

    final_accuracy = df['accuracy'].iloc[-1]
    print(f"最终检查点的准确率为: {final_accuracy:.4f}")

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "text.usetex": False,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = pd.DataFrame({
        'Test': ['Control Word Rejection'],
        'Accuracy': [final_accuracy]
    })

    barplot = sns.barplot(
        x='Test',
        y='Accuracy',
        data=plot_data,
        palette=['#2980b9'],
        ax=ax,
        edgecolor='black',
        linewidth=1.5
    )

    for p in barplot.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            xytext=(0, 8),
            textcoords='offset points',
            fontsize=16,
            weight='bold',
            color='black'
        )

    ax.set_ylabel("Final Judgment Accuracy", fontsize=14, fontweight='bold')
    ax.set_xlabel("")

    ax.set_ylim(0, 1.05)

    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

    ax.set_title(PLOT_TITLE, fontsize=18, pad=25, weight='bold')

    plt.tight_layout(pad=3.0)

    plt.savefig(OUTPUT_PLOT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"图表已保存至: {OUTPUT_PLOT_FILE}")


if __name__ == "__main__":
    try:
        accuracy_df = pd.read_csv(INPUT_CSV_FILE)
        print(f"成功从 '{INPUT_CSV_FILE}' 加载数据。")
    except FileNotFoundError:
        print(f"错误：未找到文件 '{INPUT_CSV_FILE}'。")
        exit()

    if not accuracy_df.empty:
        plot_final_accuracy(accuracy_df)
  