import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from myDB import SimpleVectorDB
# python /home/ma-user/work/mindnlp_project/build_simple_vector_db.py \
#   --meta /home/ma-user/work/mindnlp_project/faiss_db/faiss_meta.json \
#   --out /home/ma-user/work/mindnlp_project/vector_store \
#   --model all-MiniLM-L6-v2


def build_db(meta_file: str, output_dir: str, model_name: str, metric: str = "cosine"):
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    texts = []
    payloads = []
    for item in meta:
        if "药品名称" in item or "通用名称" in item:
            name = item.get("药品名称", "") or item.get("通用名称", "")
            desc_parts = [
                item.get("适应症", ""),
                item.get("用法用量", ""),
                item.get("禁忌", ""),
                item.get("注意事项", ""),
            ]
            t = " ".join([name] + [x for x in desc_parts if x])
        else:
            t = f"{item.get('full_name', '')} {item.get('indication', '')} {item.get('core_benefit', '')} {item.get('category', '')}"
        texts.append(t)
        payloads.append(item)
        
    print(f"Generating embeddings for {len(texts)} items using {model_name}...")
    embedder = SentenceTransformer(model_name)
    embs = embedder.encode(texts, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)
    
    db = SimpleVectorDB(dim=embs.shape[1], storage_dir=output_dir, metric=metric)
    db.upsert(embs, payloads)
    db.save()
    return db.count()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="/home/ma-user/work/mindnlp_project/inventory.json")
    parser.add_argument("--out", type=str, default="/home/ma-user/work/mindnlp_project/vector_store")
    parser.add_argument("--model", type=str, default="moka-ai/m3e-base")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = build_db(args.meta, args.out, args.model, args.metric)
    print(f"Built Optimized SimpleVectorDB at {args.out}, entries={n}")

if __name__ == "__main__":
    main()
