好的，这是移除所有注释并将控制台输出文本修改为更通用中文的版本。脚本的其余部分，包括所有变量和逻辑，均未作任何修改。 code Pythondownloadcontent_copyexpand_less    import os
import re
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_NAME = "distilbert-base-uncased"
TRAIN_FILE = "train.txt"

NEW_TOKENS = [
    "prallut", "vrigorn", "broukth", "rilthex", "vodrany", "Kthwo",
    "prallutik", "vrigornik", "prallutul", "vrigornul",
    "prallutov", "vrigornov", "broukthim", "rilthexim",
    "brouktheph", "rilthexeph"
]
NEW_TOKENS = list(set(NEW_TOKENS))

OUTPUT_DIR = "./wug_test_results"
NUM_TRAIN_EPOCHS = 70
SAVE_STEPS = 50
LOGGING_STEPS = 50
MLM_PROBABILITY = 0.15

class MemoryOptimizedTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"检查点已在步骤 {self.state.global_step} 保存至 {output_dir}。已清理内存。")

def setup_model_and_tokenizer(model_name, new_tokens):
    print(f"正在从 '{model_name}' 加载模型和分词器。")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    print(f"原始分词器词汇表大小: {len(tokenizer)}")
    num_added_toks = tokenizer.add_tokens(new_tokens)
    print(f"已添加 {num_added_toks} 个新词元。")
    print(f"新分词器词汇表大小: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def load_and_prepare_dataset(tokenizer, train_file):
    print(f"正在从 '{train_file}' 加载数据集。")
    try:
        dataset = load_dataset('text', data_files={'train': train_file})
    except FileNotFoundError:
        print(f"错误：未找到训练文件 '{train_file}'。")
        exit()
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    print("正在对数据集进行分词。")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=1
    )
    return tokenized_dataset['train']


if __name__ == "__main__":
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"检测到 {device_count} 个可用的GPU。")
        for i in range(device_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        print("训练将使用GPU进行。\n")
    else:
        print("警告：未检测到可用的GPU。")
        print("训练将在CPU上进行，这可能会很慢。请检查您的CUDA驱动和PyTorch安装。\n")

    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, NEW_TOKENS)
    train_dataset = load_and_prepare_dataset(tokenizer, TRAIN_FILE)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY
    )
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=8,
        save_steps=SAVE_STEPS,
        save_total_limit=None,
        logging_steps=LOGGING_STEPS,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
    )
    trainer = MemoryOptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\n开始训练。")
    
    trainer.train()
    
    print("\n训练完成。")
    print(f"模型和检查点已保存到: {OUTPUT_DIR}")
  