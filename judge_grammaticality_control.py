import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

CHECKPOINTS_DIR = "./wug_test_results"
TEST_PAIRS_FILE = "test_pairs_control.csv"
OUTPUT_RESULTS_FILE = "grammaticality_results_control.csv"

def find_sorted_checkpoints(checkpoints_dir: str) -> list:
    print(f"扫描目录 '{checkpoints_dir}' 中的检查点...")
    if not os.path.isdir(checkpoints_dir):
        print(f"错误：找不到目录 '{checkpoints_dir}'。")
        return []

    dir_contents = os.listdir(checkpoints_dir)
    checkpoints = []
    pattern = re.compile(r"checkpoint-(\d+)")

    for item in dir_contents:
        full_path = os.path.join(checkpoints_dir, item)
        if os.path.isdir(full_path):
            match = pattern.match(item)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, full_path))
    
    checkpoints.sort(key=lambda x: x[0])
    sorted_paths = [path for step, path in checkpoints]
    print(f"已找到并排序 {len(sorted_paths)} 个检查点。")
    return sorted_paths

def get_sentence_loss(model, tokenizer, sentence: str, device) -> float:
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    
    input_ids = inputs.input_ids
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
    
    return loss.item()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    try:
        test_pairs_df = pd.read_csv(TEST_PAIRS_FILE)
        print(f"已从 '{TEST_PAIRS_FILE}' 加载 {len(test_pairs_df)} 个测试对。")
    except FileNotFoundError:
        print(f"错误：找不到测试文件 '{TEST_PAIRS_FILE}'。")
        exit()

    checkpoint_paths = find_sorted_checkpoints(CHECKPOINTS_DIR)
    if not checkpoint_paths:
        print("未找到检查点，程序即将退出。")
        exit()

    all_results = []
    
    print("\n开始测试")
    for checkpoint_path in tqdm(checkpoint_paths, desc="处理检查点中"):
        
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_path).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        match = re.search(r"checkpoint-(\d+)", checkpoint_path)
        current_step = int(match.group(1)) if match else 0

        for _, row in test_pairs_df.iterrows():
            probe_name = row['probe_name']
            good_sentence = row['good_sentence']
            bad_sentence = row['bad_sentence']

            good_loss = get_sentence_loss(model, tokenizer, good_sentence, device)
            bad_loss = get_sentence_loss(model, tokenizer, bad_sentence, device)

            is_correct = 1 if good_loss < bad_loss else 0
            
            all_results.append({
                "step": current_step,
                "probe_name": probe_name,
                "is_correct": is_correct,
            })

    print("\n--- 处理结束 ---")

    results_df = pd.DataFrame(all_results)
    
    print("正在计...")
    accuracy_df = results_df.groupby(['step', 'probe_name'])['is_correct'].mean().reset_index()
    accuracy_df = accuracy_df.rename(columns={'is_correct': 'accuracy'})

    accuracy_df.to_csv(OUTPUT_RESULTS_FILE, index=False)
    
    print(f"\n结果已保存到 '{OUTPUT_RESULTS_FILE}'。")