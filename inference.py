import os
import glob
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MINDNLP_BACKEND"] = "ms"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import mindspore as ms
from mindnlp.transformers import AutoModelForCausalLM, Qwen2Tokenizer
from mindnlp.peft import PeftModel
from sentence_transformers import SentenceTransformer
from myDB import SimpleVectorDB

# ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
# =================配置区域=================
# 1. 原始基座模型
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LORA_DIR = "/root/mindnlp_project2/output_qwen_lora_inventory2"
SIMPLE_DB_DIR = "/root/mindnlp_project2/vector_store2"
EMBED_MODEL_NAME = "moka-ai/m3e-base"
# =========================================

print("⏳ 正在加载 Tokenizer...")
tokenizer = Qwen2Tokenizer.from_pretrained(BASE_MODEL_ID)

print("⏳ 正在加载基座模型 (FP16)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype="float16"
)

print("⏳ 正在加载向量库...")
vdb = None
meta_path = os.path.join(SIMPLE_DB_DIR, "meta.json")
vec_path = os.path.join(SIMPLE_DB_DIR, "vectors.npy")
if os.path.exists(meta_path) and os.path.exists(vec_path):
    vdb = SimpleVectorDB.load(SIMPLE_DB_DIR)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
else:
    print("⚠️ 未找到向量库，将不使用 RAG 检索。")

print("⏳ 正在加载 LoRA 权重...")
target_lora_path = None
if os.path.exists(os.path.join(LORA_DIR, "adapter_config.json")):
    target_lora_path = LORA_DIR
else:
    checkpoints = glob.glob(os.path.join(LORA_DIR, "checkpoint-*"))
    if checkpoints:
        target_lora_path = max(checkpoints, key=os.path.getctime)

if target_lora_path is not None and os.path.exists(target_lora_path):
    print(f"⏳ 正在加载 LoRA 权重: {target_lora_path} ...")
    model = PeftModel.from_pretrained(model, target_lora_path)
else:
    print("⚠️ 未找到 LoRA 权重，将使用基座模型推理。")

model = model.to('npu:0')
model.set_train(False)

def retrieve_knowledge(query, top_k=3):
    if vdb is None:
        return ""
    def _embed(text: str):
        vec = embed_model.encode([text], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)
    results = vdb.search_text(query, _embed, top_k=top_k)
    parts = []
    for i, r in enumerate(results, 1):
        payload = r["payload"]
        name = payload.get("药品名称") or payload.get("通用名称") or ""
        indication = payload.get("适应症", "")
        usage = payload.get("用法用量", "")
        contraind = payload.get("禁忌", "")
        notes = payload.get("注意事项", "")
        content = (
            f"药品名称：{name}\n"
            f"适应症：{indication}\n"
            f"用法用量：{usage}\n"
            f"禁忌：{contraind}\n"
            f"注意事项：{notes}"
        )
        parts.append(f"[药品 {i}]:\n{content}")
    return "\n---\n".join(parts)

def generate_rag_response(query):
    """
    构造符合训练格式的 Prompt 并生成回答
    """
    retrieved_context = retrieve_knowledge(query, top_k=3)
    if not retrieved_context:
        retrieved_context = "（无相关药品信息）"
    rag_prompt = f"请参考以下药品信息回答问题：\n### 参考信息开始 ###\n{retrieved_context}\n### 参考信息结束 ###\n\n用户的具体问题是：\n{query}"

    print("\n===== RAG Prompt (Training Format) =====")
    print(rag_prompt)
    print("===== End RAG Prompt =====\n")

    # 2. 构造完整的对话消息
    messages = [
        # System Prompt 最好与训练时保持一致
        {"role": "system", "content": "你是一个无人售药机的智能药师助手。请严格基于给定的【参考资料】回答用户问题。如果资料中提及禁忌症或库存缺货，必须发出警告或拒绝。"},
        {"role": "user", "content": rag_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="ms", padding=True)

    generated_ids = model.generate(
        model_inputs.input_ids.to('npu:0'),
        attention_mask=model_inputs.attention_mask.to('npu:0'),
        max_new_tokens=256,
        do_sample=True,
        top_k=20,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

if __name__ == "__main__":
    user_query_1 = "我牙疼得厉害，这机器里有药能治吗？"
    print("-" * 40)
    print(f"用户问题: {user_query_1}")
    print("🤖 模型回答 (Thinking...):")
    ans1 = generate_rag_response(user_query_1)
    print(ans1)

    user_query_2 = "我是孕妇，牙疼，能吃这个布洛芬吗？"
    print("-" * 40)
    print(f"用户问题: {user_query_2}")
    print("🤖 模型回答 (Thinking...):")
    ans2 = generate_rag_response(user_query_2)
    print(ans2)

    user_query_3 = "我想买个创可贴。"
    print("-" * 40)
    print(f"用户问题: {user_query_3}")
    print("🤖 模型回答 (Thinking...):")
    ans3 = generate_rag_response(user_query_3)
    print(ans3)
