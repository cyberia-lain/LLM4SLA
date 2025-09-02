import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import os
import re

FINAL_CHECKPOINT_PATH = "./wug_test_results/checkpoint-6580"
OUTPUT_PLOT_FILE_ALL = "Attention_All_Heads.png"
OUTPUT_PLOT_FILE_FOCUSED = "Attention_Syntax_Head_Focused.png"

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'

TEST_SENTENCE = "The scientist who studies the broukthim prallutik ."

def visualize_attention():

    print("开始分析注意力...")
    
    try:
        model = AutoModelForMaskedLM.from_pretrained(FINAL_CHECKPOINT_PATH, output_attentions=True)
        tokenizer = AutoTokenizer.from_pretrained(FINAL_CHECKPOINT_PATH)
    except OSError:
        print(f"错误：在 '{FINAL_CHECKPOINT_PATH}' 路径下找不到模型。")
        return

    model.eval()

    inputs = tokenizer(TEST_SENTENCE, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    attentions = outputs.attentions 
    
    try:
        verb_token_index = tokens.index('prallutik')
        subject_token_index = tokens.index('scientist')
        distractor_token_index = tokens.index('broukthim')
    except ValueError as e:
        print(f"错误：句子中缺少关键的词  {e}")
        print(f"分词结果是: {tokens}")
        return

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 2.5, num_layers * 2.5))
    fig.suptitle("Attention Patterns Across All Layers and Heads", fontsize=18, fontweight='bold', color='black')

    for layer in range(num_layers):
        for head in range(num_heads):
            attention_head = attentions[layer][0, head].cpu().numpy()
            sns.heatmap(
                attention_head,
                ax=axes[layer, head],
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                cmap="viridis"
            )
            axes[layer, head].set_title(f"L{layer} H{head}", fontsize=12, color='black')
            axes[layer, head].scatter(subject_token_index, verb_token_index, color='red', s=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PLOT_FILE_ALL, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"所有注意力头的概览图已经保存到: {OUTPUT_PLOT_FILE_ALL}")

    print("\n正在寻找连接动词和主语的注意力头...")
    best_head_info = {"layer": -1, "head": -1, "score": -1}

    for layer in range(num_layers):
        for head in range(num_heads):
            attention_head = attentions[layer][0, head]
            attention_to_subject = attention_head[verb_token_index, subject_token_index].item()
            attention_to_distractor = attention_head[verb_token_index, distractor_token_index].item()
            score = attention_to_subject / (attention_to_distractor + 1e-6)
            
            if score > best_head_info["score"]:
                best_head_info = {"layer": layer, "head": head, "score": score}

    print(f"找到的最佳注意力头: 层 {best_head_info['layer']}, 头 {best_head_info['head']} (得分: {best_head_info['score']:.2f})")

    best_attention_head = attentions[best_head_info['layer']][0, best_head_info['head']].cpu().numpy()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        best_attention_head, 
        xticklabels=tokens, 
        yticklabels=tokens, 
        cmap="viridis",
        linewidths=.5,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 16, "weight": "bold", "color": "black"}
    )

    plt.title(
        f"Attention Propagation in Long-Distance Dependency Resolution: (Layer {best_head_info['layer']}, Head {best_head_info['head']})",
        fontsize=18,
        fontweight='bold',
        color='black',
        pad=20
    )
    plt.xlabel("Key (Words being attended to)", fontsize=16, color='black')
    plt.ylabel("Query (Words doing the attending)", fontsize=14, fontweight='bold', color='black')

    plt.xticks(rotation=45, ha="right", fontsize=13, color='black')
    plt.yticks(rotation=0, fontsize=13, color='black')
    
    plt.savefig(OUTPUT_PLOT_FILE_FOCUSED, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图已经保存到: {OUTPUT_PLOT_FILE_FOCUSED}")


if __name__ == "__main__":
    visualize_attention()