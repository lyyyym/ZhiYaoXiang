import os
import json
import mindspore
# 显式导入 Qwen2Tokenizer，不再用 AutoTokenizer
from mindnlp.transformers import AutoModelForCausalLM, Qwen2Tokenizer
from mindnlp.peft import LoraConfig, get_peft_model, TaskType
from mindnlp.transformers import TrainingArguments, Trainer 

# 消除 Tokenizers 并行警告 (防止死锁)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 配置参数
# ==========================================
model_name_or_path = "Qwen/Qwen2.5-7B-Instruct" 
# 确保这里指向的是你生成的高质量数据文件
data_path = "/root/mindnlp_project2/new/medical_rag_finetune_data.jsonl"
output_dir = "./output_qwen_lora_inventory2"

# ==========================================
# 2. 加载模型和分词器
# ==========================================
print(">>> Loading Tokenizer...")
# !!! 关键修改 2: 使用 Qwen2Tokenizer 直接加载，绕过 AttributeError !!!
try:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path)
except Exception as e:
    print(f"⚠️ Qwen2Tokenizer 加载失败，尝试使用 AutoTokenizer: {e}")
    from mindnlp.transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token 

print(">>> Loading Model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, 
    ms_dtype=mindspore.float16 
)

# ==========================================
# 3. 注入 LoRA
# ==========================================
print(">>> Applying LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==========================================
# 4. 数据处理
# ==========================================
class DrugDataset:
    def __init__(self, path, tokenizer, max_len=512):
        self.data = []
        # 检查文件是否存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ 数据文件 {path} 不存在，请先运行 generate_from_manuals.py")
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except:
                    continue
        print(f">>> 已加载训练数据: {len(self.data)} 条")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        item = self.data[index]
        instruction = item.get('instruction', '')
        user_input = item.get('input', '')
        output = item.get('output', '')
        
        # 构造符合 Qwen 格式的 Prompt
        text = f"User: {instruction}\n{user_input}\nAssistant: {output}"
        
        # Tokenize
        encodings = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True,
            return_tensors="ms"
        )
        
        # 返回字典给 Trainer
        return {
            "input_ids": encodings['input_ids'].squeeze(),
            "attention_mask": encodings['attention_mask'].squeeze(),
            "labels": encodings['input_ids'].squeeze()
        }

    def __len__(self):
        return len(self.data)

# 实例化数据集
train_dataset = DrugDataset(data_path, tokenizer)

# ==========================================
# 5. 训练参数与开始
# ==========================================
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    fp16=False,  # 保持 False
    per_device_train_batch_size=1,
    dataloader_drop_last=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print(">>> Starting Training...")
trainer.train()