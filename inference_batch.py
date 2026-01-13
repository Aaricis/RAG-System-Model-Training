import argparse
import faiss
import gc
import json
import os
import sqlite3

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import get_inference_user_prompt, get_inference_system_prompt, parse_generated_answer

# load_dotenv()
# hf_token = os.getenv("hf_token")
# login(token=hf_token)

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_folder", type=str, default="./data")
argparser.add_argument("--passage_file", type=str, default="corpus.txt")
argparser.add_argument("--index_folder", type=str, default="./vector_database")
argparser.add_argument("--index_file", type=str, default="passage_index.faiss")
argparser.add_argument("--sqlite_file", type=str, default="passage_store.db")
argparser.add_argument("--test_data_path", type=str, default="./data/test_open.txt")
argparser.add_argument("--qrels_path", type=str, default="./data/qrels.txt")
argparser.add_argument("--retriever_model_path", type=str, default="")
argparser.add_argument("--reranker_model_path", type=str, default="")
argparser.add_argument("--generator_model", type=str, default="Qwen/Qwen3-1.7B")
argparser.add_argument("--result_file_name", type=str, default="result.json")
args = argparser.parse_args()

data_folder = args.data_folder
passage_file = args.passage_file
index_folder = args.index_folder
index_file = args.index_file
sqlite_file = args.sqlite_file
test_data_path = args.test_data_path
retriever_model_path = args.retriever_model_path
reranker_model_path = args.reranker_model_path
qrels_path = args.qrels_path
result_file = args.result_file_name

###############################################################################
# 0. parameters
TOP_K = 10
TOP_M = 3  # top M passages send to LLM
GEN_MAXLEN = 1280
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_Q = 16  # you can adjust this value to fit hardware constraints
BATCH_GEN = 2  # you can adjust this value to fit hardware constraints
TEST_DATA_SIZE = -1  # set to -1 to run on the whole test set

###############################################################################
# 1. load db and index

sqlite_path = f"{index_folder}/{sqlite_file}"
conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()

retriever = SentenceTransformer(retriever_model_path, device=DEVICE)
vram_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]
print(f"Retriever VRAM: {vram_allocated / 1e9:.2f} GB")

index = faiss.read_index(os.path.join(index_folder, index_file))


###############################################################################
# 2. load dataset
def load_qrels_gold_pids(qrels_path):
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    qid2gold = {}
    for qid, pid2lab in qrels.items():
        gold = {pid for pid, lab in pid2lab.items() if str(lab) != "0"}
        qid2gold[qid] = gold
    return qid2gold


tests = []  # list of dict: {qid, query, gold_answer, gold_pids(set)}
qid2gold = load_qrels_gold_pids(qrels_path)

with open(test_data_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        qid = obj.get("qid")
        query = obj.get("rewrite")
        gold_answer = (obj.get("answer")).get("text", "")
        gold_pids = qid2gold.get(qid, set())
        tests.append({"qid": qid, "query": query, "gold_answer": gold_answer, "gold_pids": gold_pids})

tests = tests[:TEST_DATA_SIZE]


# =========================
# 3. eval metrices（Recall / MRR）
# =========================

def recall_at_k(retrieved_pids, gold_pids, k):
    topk = retrieved_pids[:k]
    return 1.0 if any(pid in gold_pids for pid in topk) else 0.0


def mrr_at_k(reranked_pids, gold_pids, k):
    for rank, pid in enumerate(reranked_pids[:k]):
        if pid in gold_pids:
            score = 1.0 / (rank + 1)
            return score
    return 0.0


# =========================
# 4. inference
# =========================

reranker = CrossEncoder(reranker_model_path, device=DEVICE)
print(f"Reranker VRAM: {(torch.cuda.memory_stats()['allocated_bytes.all.current'] - vram_allocated) / 1e9:.2f} GB")
vram_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]

# model_id = args.generator_model
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

# 本地模型目录路径（绝对或相对路径）
model_path = "/mnt/data/models/Qwen3-1.7B"  # 替换为你的实际路径

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True  # 如果模型含自定义代码（如Qwen）必须加
)

# 加载模型（修正了dtype参数名错误）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True  # 与tokenizer保持一致
)

print(f"LLM VRAM: {(torch.cuda.memory_stats()['allocated_bytes.all.current'] - vram_allocated) / 1e9:.2f} GB")
vram_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]

R_at_K_sum = 0.0
MRR_sum = 0.0
N = 0

output_records = []
for b_start in tqdm(range(0, len(tests), BATCH_Q)):
    batch = tests[b_start:b_start + BATCH_Q]
    qids = [ex["qid"] for ex in batch]
    queries = [ex["query"] for ex in batch]
    gold_sets = [ex["gold_pids"] for ex in batch]
    gold_ans = [ex["gold_answer"] for ex in batch]

    # 1) retrieve and search from FAISS
    prefix_queries = ["query: " + q for q in queries]  # adjust the query prefix for retriever
    q_embs = retriever.encode(
        prefix_queries, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=BATCH_Q
    )
    D, I = index.search(q_embs,
                        TOP_K)  # D/I shape = (BATCH_Q, TOP_K), D is distances between queries and corpus, I is rowids of the retrieved passages

    # 2) retrieve all passages needed in this batch
    need_rowids = set(int(rid) for row in I for rid in row.tolist())
    placeholders = ",".join(["?"] * len(need_rowids)) or "NULL"
    sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
    rows = cur.execute(sql, tuple(need_rowids)).fetchall()
    rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}

    # 3) create（cand_ids, cand_texts）of each query and calculate recall@K
    batch_cand_ids, batch_cand_texts = [], []
    for b, row in enumerate(I):
        rid_list = row.tolist()
        cand_ids, cand_texts = [], []
        for rid in rid_list:
            tup = rowid2pt.get(int(rid))
            if tup is None: continue
            pid, text = tup
            cand_ids.append(pid)
            cand_texts.append(text)
        batch_cand_ids.append(cand_ids)
        batch_cand_texts.append(cand_texts)
        R_at_K_sum += recall_at_k(cand_ids, gold_sets[b], TOP_K)

    # 4) create (query, passage) pairs into flat list for reranker.predict
    flat_pairs = []
    idx_slices = []  # index slices to recover each query's pairs
    cursor = 0
    for q, ctexts in zip(queries, batch_cand_texts):
        n = len(ctexts)
        if n == 0:
            idx_slices.append((cursor, cursor))
            continue
        flat_pairs.extend(zip([q] * n, ctexts))
        idx_slices.append((cursor, cursor + n))
        cursor += n

    # if no candidates in this batch, skip to next batch
    if len(flat_pairs) == 0:
        MRR_sum += 0.0 * len(batch)
        N += len(batch)
        continue

    # 5) reranker score all pairs in flat list
    flat_scores = reranker.predict(flat_pairs)

    # 6) recover each query, sort by reranker score, calculate MRR@K, create generation prompt
    messages_list = []
    rerank_info_list = []
    for b, (q, (low, high)) in enumerate(zip(queries, idx_slices)):
        if low == high:  # if no candidates for this query
            MRR_sum += 0.0
            N += 1
            messages_list.append(None)
            rerank_info_list.append(None)
            continue

        scores = flat_scores[low:high]
        cand_ids = batch_cand_ids[b]
        cand_text = batch_cand_texts[b]
        reranked = sorted(zip(scores, cand_ids, cand_text), key=lambda x: x[0], reverse=True)
        reranked_pids = [pid for _, pid, _ in reranked]
        MRR_sum += mrr_at_k(reranked_pids, gold_sets[b], TOP_K)
        context_list = [text for _, _, text in reranked]
        context_list = context_list[:min(TOP_M, len(context_list))]
        messages = [
            {
                "role": "system",
                "content": get_inference_system_prompt()
            },
            {
                "role": "user",
                "content": get_inference_user_prompt(queries[b], context_list)
            }
        ]
        messages_list.append(messages)
        rerank_info_list.append([
            {"pid": pid, "text": text, "score": float(score)}
            for score, pid, text in reranked
        ])
    # 7) generation for all queries in this batch
    pending = [(idx, passage) for idx, passage in enumerate(messages_list) if passage is not None]
    for g_start in range(0, len(pending), BATCH_GEN):
        chunk = pending[g_start:g_start + BATCH_GEN]
        idxs, msgs_batch = zip(*chunk)
        tokenizer.padding_side = "left"
        rendered_prompts = [
            tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
            for m in msgs_batch
        ]

        inputs = tokenizer(
            rendered_prompts, padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=GEN_MAXLEN)
        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
        for j, ans in zip(idxs, decoded):
            pred_ans = parse_generated_answer(ans.strip())
            N += 1
            output_records.append({
                "qid": qids[j],
                "query": queries[j],
                "retrieved": rerank_info_list[j],
                "generated": pred_ans,
                "gold_answer": gold_ans[j],
                "faiss_distances": D[j].tolist(),
                "faiss_rowids": I[j].tolist(),
                "gold_pids": list(gold_sets[j])
            })

del model
del retriever
del reranker
gc.collect()
torch.cuda.empty_cache()

res = [record["generated"] for record in output_records]
ans = [record["gold_answer"] for record in output_records]

# sentence_scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
sentence_scorer = SentenceTransformer(
    "/root/autodl-tmp/RAG-System-Model-Training/all-MiniLM-L6-v2",
    device=DEVICE
)
emb_res = sentence_scorer.encode(res, convert_to_tensor=True, normalize_embeddings=True)
emb_gold = sentence_scorer.encode(ans, convert_to_tensor=True, normalize_embeddings=True)
scores = util.cos_sim(emb_res, emb_gold)
diag_scores = scores.diag().tolist()
bi_encoder_similarity = np.mean(diag_scores)
# =========================
# 5. results
# =========================
print(f"Queries evaluated: {N}")
print(f"Recall@{TOP_K}: {R_at_K_sum / max(N, 1):.4f}")
print(f"MRR@{TOP_K} (after rerank): {MRR_sum / max(N, 1):.4f}")
print(f"Bi-Encoder CosSim: {bi_encoder_similarity:.4f}")

final = {"data_size": N,
         f"recall@{TOP_K}": R_at_K_sum / max(N, 1),
         f"mrr@{TOP_K}": MRR_sum / max(N, 1),
         "Bi-Encoder_CosSim": bi_encoder_similarity,
         "records": output_records}

os.makedirs("results", exist_ok=True)
result_file_path = os.path.join("results", result_file)
with open(result_file_path, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)
