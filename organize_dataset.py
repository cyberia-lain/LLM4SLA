import os
import random

SOURCE_DATA_DIR = "source_data"
TEST_DATA_DIR = "test"

OUTPUT_TRAIN_FILE = "train.txt"
OUTPUT_TEST_FILE = "test_pairs.csv"

def process_training_files(source_dir, output_file):

    print("正在处理训练文件...")
    
    if not os.path.isdir(source_dir):
        print(f"错误：找不到训练数据目录 '{source_dir}'。")
        return

    all_training_sentences = []
    source_filenames = os.listdir(source_dir)
    
    print(f"发现 {len(source_filenames)} 个训练源文件。")

    for filename in source_filenames:
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(source_dir, filename)
        print(f"  正在读取文件: {filename}")
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
            all_training_sentences.extend(sentences)

    print(f"\n总共读取了 {len(all_training_sentences)} 条训练句子。")
    
    print("正在打乱训练句子顺序...")
    random.shuffle(all_training_sentences)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_training_sentences))
        
    print(f"训练数据处理完成，已保存到 '{output_file}'。")


def process_testing_files(test_dir, output_file):

    print("\n正在处理测试文件...")

    if not os.path.isdir(test_dir):
        print(f"错误：找不到测试数据目录 '{test_dir}'。")
        return

    all_test_pairs = []
    test_filenames = os.listdir(test_dir)
    
    print(f"发现 {len(test_filenames)} 个测试源文件。")

    for filename in test_filenames:
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(test_dir, filename)
        print(f"  正在读取文件: {filename}")
        with open(filepath, 'r', encoding='utf-8') as f:
            pairs = [line.strip() for line in f if line.strip()]
            all_test_pairs.extend(pairs)
            
    print(f"\n总共读取了 {len(all_test_pairs)} 个测试对。")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("probe_name,good_sentence,bad_sentence\n")
        f.write('\n'.join(all_test_pairs))
        
    print(f"处理完成，已保存到 '{output_file}'。")


if __name__ == "__main__":
    process_training_files(SOURCE_DATA_DIR, OUTPUT_TRAIN_FILE)
    
    process_testing_files(TEST_DATA_DIR, OUTPUT_TEST_FILE)
    
    print("\n数据整理完成。")```