import argparse
import gc
import json
import os
import sqlite3

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from stable_baselines3 import PPO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_inference_user_prompt, get_inference_system_prompt, parse_generated_answer
from collections import Counter
import matplotlib.pyplot as plt

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
argparser.add_argument("--rl_model_path", type=str, default="./output/rl.zip")
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

# Set OMP_NUM_THREADS to a valid number
os.environ["OMP_NUM_THREADS"] = "4"

###############################################################################
# 0. parameters
TOP_K = 10
GEN_MAXLEN = 1280
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_Q = 64
BATCH_GEN = 2
TEST_DATA_SIZE = -1

###############################################################################
# Observation builder (vectorized for batch)
###############################################################################

def build_obs_single(reranked_data, llm_tokenizer, query):
    """Build a single observation vector from reranked data."""
    scores_list = np.array([float(s) for s, _, _ in reranked_data], dtype=np.float32)
    if len(scores_list) == 0:
        scores_list = np.zeros(TOP_K, dtype=np.float32)
    if len(scores_list) < TOP_K:
        pad = np.ones(TOP_K - len(scores_list), dtype=np.float32) * scores_list.min()
        scores_list = np.concatenate([scores_list, pad])
    scores = scores_list[:TOP_K]

    mu = scores.mean()
    std = scores.std() + 1e-6
    scores_norm = (scores - mu) / std
    scores_norm = np.clip(scores_norm, -3, 3)

    gap12 = scores_norm[0] - scores_norm[1] if TOP_K > 1 else 0.0
    gap1k = scores_norm[0] - scores_norm[-1]

    p = np.exp(scores_norm - scores_norm.max())
    p = p / (p.sum() + 1e-8)
    p = np.clip(p, 1e-8, None)
    entropy = -np.sum(p * np.log(p + 1e-8))
    entropy_norm = entropy / np.log(TOP_K + 1e-8)

    gap12_norm = np.tanh(gap12)
    gap1k_norm = np.tanh(gap1k)

    q_tokens = llm_tokenizer.encode(query, add_special_tokens=False)
    MAX_Q_LEN = 64
    query_len_norm = min(len(q_tokens), MAX_Q_LEN) / MAX_Q_LEN

    ctx_lens = [
        len(llm_tokenizer.encode(t, add_special_tokens=False))
        for _, _, t in reranked_data[:TOP_K]
    ]
    if len(ctx_lens) == 0:
        avg_ctx_len_norm = 0.0
    else:
        MAX_CTX_LEN = 256
        avg_ctx_len_norm = min(float(np.mean(ctx_lens)), MAX_CTX_LEN) / MAX_CTX_LEN

    if len(ctx_lens) > 2:
        slope = np.mean(np.diff(ctx_lens[:5]))
    else:
        slope = 0.0
    slope_norm = np.tanh(slope / 50)

    obs = np.concatenate([
        scores_norm,
        [gap12_norm, gap1k_norm, entropy_norm, query_len_norm, avg_ctx_len_norm, slope_norm]
    ]).astype(np.float32)
    return obs

def batch_infer_top_m(reranked_list, llm_tokenizer, queries, ppo_model):
    """
    Batch inference of top_m for a list of queries.

    Args:
        reranked_list: list of reranked data (one per query), each is
                       sorted list of (score, pid, text). None for skipped queries.
        llm_tokenizer: tokenizer for encoding queries/passages.
        queries: list of query strings (same length as reranked_list).
        ppo_model: loaded PPO model.

    Returns:
        top_m_results: list of int (or None for skipped queries).
    """
    # Collect valid indices and their observations
    valid_indices = []
    obs_list = []
    for i, reranked in enumerate(reranked_list):
        if reranked is not None:
            obs = build_obs_single(reranked, llm_tokenizer, queries[i])
            obs_list.append(obs)
            valid_indices.append(i)

    top_m_results = [None] * len(reranked_list)

    if len(obs_list) == 0:
        return top_m_results

    # Stack into a single batch tensor
    obs_batch = np.stack(obs_list, axis=0)  # (B, obs_dim)
    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=ppo_model.device)

    # Batched forward pass through the PPO policy
    with torch.no_grad():
        policy = ppo_model.policy
        # get_distribution handles feature extraction + action distribution
        dist = policy.get_distribution(obs_tensor)
        # Deterministic: take the mode (argmax) of the distribution
        actions = dist.mode()  # (B,)

    actions_np = actions.cpu().numpy()
    for idx, vi in enumerate(valid_indices):
        top_m_results[vi] = int(actions_np[idx]) + 1

    return top_m_results

###############################################################################
# 1. load db and index
###############################################################################

ppo_model = PPO.load(args.rl_model_path, device=DEVICE)

sqlite_path = f"{index_folder}/{sqlite_file}"
conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()

retriever = SentenceTransformer(retriever_model_path, device=DEVICE)
vram_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]
print(f"Retriever VRAM: {vram_allocated / 1e9:.2f} GB")

index = faiss.read_index(os.path.join(index_folder, index_file))

###############################################################################
# 2. load dataset
###############################################################################

def load_qrels_gold_pids(qrels_path):
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    qid2gold = {}
    for qid, pid2lab in qrels.items():
        gold = {pid for pid, lab in pid2lab.items() if str(lab) != "0"}
        qid2gold[qid] = gold
    return qid2gold

tests = []
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

def recall_at_k(retrieved_pids, gold_pids, k):
    topk = retrieved_pids[:k]
    return 1.0 if any(pid in gold_pids for pid in topk) else 0.0

def mrr_at_k(reranked_pids, gold_pids, k):
    for rank, pid in enumerate(reranked_pids[:k]):
        if pid in gold_pids:
            return 1.0 / (rank + 1)
    return 0.0

###############################################################################
# 3. load reranker + generator
###############################################################################

reranker = CrossEncoder(reranker_model_path, device=DEVICE)
print(f"Reranker VRAM: {(torch.cuda.memory_stats()['allocated_bytes.all.current'] - vram_allocated) / 1e9:.2f} GB")
vram_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]

model_path = "/mnt/data/models/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

print(f"LLM VRAM: {(torch.cuda.memory_stats()['allocated_bytes.all.current'] - vram_allocated) / 1e9:.2f} GB")
vram_allocated = torch.cuda.memory_stats()["allocated_bytes.all.current"]

###############################################################################
# 4. inference loop
###############################################################################

R_at_K_sum = 0.0
MRR_sum = 0.0
N = 0
output_records = []
top_m_counter = Counter()
top_m_values = []

for b_start in tqdm(range(0, len(tests), BATCH_Q)):
    batch = tests[b_start:b_start + BATCH_Q]
    qids = [ex["qid"] for ex in batch]
    queries = [ex["query"] for ex in batch]
    gold_sets = [ex["gold_pids"] for ex in batch]
    gold_ans = [ex["gold_answer"] for ex in batch]

    # 1) FAISS retrieval
    prefix_queries = ["query: " + q for q in queries]
    q_embs = retriever.encode(
        prefix_queries, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=BATCH_Q
    )
    D, I = index.search(q_embs, TOP_K)

    # 2) fetch passages from SQLite
    need_rowids = set(int(rid) for row in I for rid in row.tolist())
    placeholders = ",".join(["?"] * len(need_rowids)) or "NULL"
    sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
    rows = cur.execute(sql, tuple(need_rowids)).fetchall()
    rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}

    # 3) build candidates per query, compute recall@K
    batch_cand_ids, batch_cand_texts = [], []
    for b, row in enumerate(I):
        rid_list = row.tolist()
        cand_ids, cand_texts = [], []
        for rid in rid_list:
            tup = rowid2pt.get(int(rid))
            if tup is None:
                continue
            pid, text = tup
            cand_ids.append(pid)
            cand_texts.append(text)
        batch_cand_ids.append(cand_ids)
        batch_cand_texts.append(cand_texts)
        R_at_K_sum += recall_at_k(cand_ids, gold_sets[b], TOP_K)

    # 4) flatten (query, passage) pairs for reranker
    flat_pairs = []
    idx_slices = []
    cursor = 0
    for q, ctexts in zip(queries, batch_cand_texts):
        n = len(ctexts)
        if n == 0:
            idx_slices.append((cursor, cursor))
            continue
        flat_pairs.extend(zip([q] * n, ctexts))
        idx_slices.append((cursor, cursor + n))
        cursor += n

    if len(flat_pairs) == 0:
        MRR_sum += 0.0 * len(batch)
        N += len(batch)
        continue

    # 5) reranker scoring
    flat_scores = reranker.predict(flat_pairs)

    # 6) rerank per query, compute MRR, collect reranked data for PPO
    reranked_data_list = []  # parallel to batch, None if no candidates
    rerank_info_list = []
    for b, (q, (low, high)) in enumerate(zip(queries, idx_slices)):
        if low == high:
            MRR_sum += 0.0
            N += 1
            reranked_data_list.append(None)
            rerank_info_list.append(None)
            continue

        scores = flat_scores[low:high]
        cand_ids = batch_cand_ids[b]
        cand_text = batch_cand_texts[b]
        reranked = sorted(zip(scores, cand_ids, cand_text), key=lambda x: x[0], reverse=True)
        reranked_pids = [pid for _, pid, _ in reranked]
        MRR_sum += mrr_at_k(reranked_pids, gold_sets[b], TOP_K)

        reranked_data_list.append(reranked)
        rerank_info_list.append([
            {"pid": pid, "text": text, "score": float(score)}
            for score, pid, text in reranked
        ])

    # 7) *** BATCH PPO inference for top_m ***
    top_m_list = batch_infer_top_m(reranked_data_list, tokenizer, queries, ppo_model)
    # 更新统计
    for tm in top_m_list:
        if tm is None:
            continue
        top_m_counter[tm] += 1
        top_m_values.append(tm)

    # 8) build generation prompts using batched top_m
    messages_list = []
    for b, reranked in enumerate(reranked_data_list):
        if reranked is None:
            messages_list.append(None)
            continue

        top_m = top_m_list[b]
        context_list = [text for _, _, text in reranked]
        context_list = context_list[:min(top_m, len(context_list))]
        messages = [
            {"role": "system", "content": get_inference_system_prompt()},
            {"role": "user", "content": get_inference_user_prompt(queries[b], context_list)}
        ]
        messages_list.append(messages)

    # 9) LLM generation
    pending = [(idx, m) for idx, m in enumerate(messages_list) if m is not None]
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
                "gold_pids": list(gold_sets[j]),
            })

###############################################################################
# 5. cleanup + scoring
###############################################################################

del model
del retriever
del reranker
gc.collect()
torch.cuda.empty_cache()

res = [record["generated"] for record in output_records]
ans = [record["gold_answer"] for record in output_records]

sentence_scorer = SentenceTransformer(
    "/root/autodl-tmp/RAG-System-Model-Training/all-MiniLM-L6-v2",
    device=DEVICE
)
emb_res = sentence_scorer.encode(res, convert_to_tensor=True, normalize_embeddings=True)
emb_gold = sentence_scorer.encode(ans, convert_to_tensor=True, normalize_embeddings=True)
scores = util.cos_sim(emb_res, emb_gold)
diag_scores = scores.diag().tolist()
bi_encoder_similarity = np.mean(diag_scores)

print(f"Queries evaluated: {N}")
print(f"Recall@{TOP_K}: {R_at_K_sum / max(N, 1):.4f}")
print(f"MRR@{TOP_K} (after rerank): {MRR_sum / max(N, 1):.4f}")
print(f"Bi-Encoder CosSim: {bi_encoder_similarity:.4f}")

final = {
    "data_size": N,
    f"recall@{TOP_K}": R_at_K_sum / max(N, 1),
    f"mrr@{TOP_K}": MRR_sum / max(N, 1),
    "Bi-Encoder_CosSim": bi_encoder_similarity,
    "records": output_records
}

os.makedirs("results", exist_ok=True)
result_file_path = os.path.join("results", result_file)
# with open(result_file_path, "w", encoding="utf-8") as f:
#     json.dump(final, f, indent=2, ensure_ascii=False)

# 计算统计
if len(top_m_values) > 0:
    vals = np.array(top_m_values)
    stats = {
        "count_total": int(vals.size),
        "counts": dict(top_m_counter),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "percentiles": {
            "25": float(np.percentile(vals, 25)),
            "50": float(np.percentile(vals, 50)),
            "75": float(np.percentile(vals, 75)),
            "90": float(np.percentile(vals, 90))
        }
    }
    for n_docs, count in sorted(top_m_counter.items()):
        print(f"  {n_docs} 个相关文档: {count} 个 queries ({100 * count / N:.1f}%)")
else:
    stats = {"count_total": 0, "counts": {}, "mean": None}

print("PPO top_m stats:", stats)

# 保存 stats 和 records（可与最终 result 一起保存）
# with open("results/top_m_stats.json", "w", encoding="utf-8") as f:
#     json.dump(stats, f, indent=2, ensure_ascii=False)

# 可以把 top_m_counter 也并入 final json
final["top_m_stats"] = stats
with open(result_file_path, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

if len(top_m_values) > 0:
    xs = sorted(top_m_counter.keys())
    ys = [top_m_counter[x] for x in xs]
    plt.bar(xs, ys)
    plt.xlabel("top_m")
    plt.ylabel("count")
    plt.title("Distribution of PPO selected top_m")
    plt.savefig("./output/top_m_hist.png", dpi=200)
    plt.close()