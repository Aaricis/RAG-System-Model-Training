# =========================
# 0. imports（安全）
# =========================
import argparse
import gc
import json
import os
import sqlite3
import time

import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import (
    get_inference_user_prompt,
    get_inference_system_prompt,
    parse_generated_answer,
)

# 强制 HuggingFace 离线（作业环境非常重要）
os.environ["HF_HUB_OFFLINE"] = "1"

# =========================
# 1. utils
# =========================
def load_qrels_gold_pids(qrels_path):
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    qid2gold = {}
    for qid, pid2lab in qrels.items():
        gold = {pid for pid, lab in pid2lab.items() if str(lab) != "0"}
        qid2gold[qid] = gold
    return qid2gold

def recall_at_k(retrieved_pids, gold_pids, k):
    return 1.0 if any(pid in gold_pids for pid in retrieved_pids[:k]) else 0.0

def mrr_at_k(reranked_pids, gold_pids, k):
    for rank, pid in enumerate(reranked_pids[:k]):
        if pid in gold_pids:
            return 1.0 / (rank + 1)
    return 0.0

# =========================
# 2. main RAG pipeline
# =========================
def run_rag(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TOP_K = 10
    TOP_M = 3
    GEN_MAXLEN = 1280
    BATCH_Q = 16
    BATCH_GEN = 2

    # -------------------------
    # load DB / FAISS
    # -------------------------
    conn = sqlite3.connect(os.path.join(args.index_folder, args.sqlite_file))
    cur = conn.cursor()

    index = faiss.read_index(
        os.path.join(args.index_folder, args.index_file)
    )

    # -------------------------
    # load retriever / reranker
    # -------------------------
    retriever = SentenceTransformer(
        args.retriever_model_path,
        device=DEVICE
    )

    reranker = CrossEncoder(
        args.reranker_model_path,
        device=DEVICE
    )

    # -------------------------
    # load test data
    # -------------------------
    qid2gold = load_qrels_gold_pids(args.qrels_path)
    tests = []

    with open(args.test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj["qid"]
            tests.append({
                "qid": qid,
                "query": obj["rewrite"],
                "gold_answer": obj["answer"]["text"],
                "gold_pids": qid2gold.get(qid, set())
            })

    # -------------------------
    # vLLM
    # -------------------------
    model_path = "/mnt/data/models/Qwen3-1.7B"

    # llm = LLM(
    #     model=model_path,
    #     tokenizer=model_path,
    #     trust_remote_code=True,
    #     dtype="auto",
    #     gpu_memory_utilization=0.90,
    # )

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.75,
        max_num_seqs=8,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=GEN_MAXLEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    # -------------------------
    # metrics
    # -------------------------
    R_sum, MRR_sum, N = 0.0, 0.0, 0
    output_records = []

    # =========================
    # inference
    # =========================
    for b_start in tqdm(range(0, len(tests), BATCH_Q)):
        batch = tests[b_start:b_start+BATCH_Q]
        qids = [b["qid"] for b in batch]
        queries = [b["query"] for b in batch]
        gold_sets = [b["gold_pids"] for b in batch]
        gold_answers = [b["gold_answer"] for b in batch]

        # ---- retrieve ----
        q_embs = retriever.encode(
            ["query: " + q for q in queries],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        D, I = index.search(q_embs, TOP_K)

        # ---- fetch passages ----
        need_rowids = set(int(rid) for row in I for rid in row.tolist())
        placeholders = ",".join(["?"] * len(need_rowids))
        rows = cur.execute(
            f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})",
            tuple(need_rowids),
        ).fetchall()

        rowid2pt = {rid: (pid, text) for rid, pid, text in rows}

        batch_cand_ids, batch_cand_texts = [], []
        for b, row in enumerate(I):
            cand_ids, cand_texts = [], []
            for rid in row.tolist():
                if rid in rowid2pt:
                    pid, text = rowid2pt[rid]
                    cand_ids.append(pid)
                    cand_texts.append(text)
            batch_cand_ids.append(cand_ids)
            batch_cand_texts.append(cand_texts)
            R_sum += recall_at_k(cand_ids, gold_sets[b], TOP_K)

        # ---- rerank ----
        flat_pairs, slices = [], []
        cur_idx = 0
        for q, texts in zip(queries, batch_cand_texts):
            flat_pairs.extend([(q, t) for t in texts])
            slices.append((cur_idx, cur_idx + len(texts)))
            cur_idx += len(texts)

        flat_scores = reranker.predict(flat_pairs)

        messages_list = []
        rerank_infos = []

        for b, (lo, hi) in enumerate(slices):
            scores = flat_scores[lo:hi]
            cand_ids = batch_cand_ids[b]
            cand_texts = batch_cand_texts[b]

            reranked = sorted(
                zip(scores, cand_ids, cand_texts),
                key=lambda x: x[0],
                reverse=True
            )

            reranked_pids = [pid for _, pid, _ in reranked]
            MRR_sum += mrr_at_k(reranked_pids, gold_sets[b], TOP_K)

            context = [t for _, _, t in reranked[:TOP_M]]

            messages_list.append([
                {"role": "system", "content": get_inference_system_prompt()},
                {"role": "user", "content": get_inference_user_prompt(queries[b], context)}
            ])

            rerank_infos.append([
                {"pid": pid, "text": text, "score": float(score)}
                for score, pid, text in reranked
            ])

        # ---- generation ----
        for g_start in range(0, len(messages_list), BATCH_GEN):
            chunk = messages_list[g_start:g_start+BATCH_GEN]
            idxs = list(range(g_start, g_start + len(chunk)))

            prompts = [
                tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for m in chunk
            ]

            outs = llm.generate(prompts, sampling_params)
            answers = [o.outputs[0].text for o in outs]

            for i, ans in zip(idxs, answers):
                pred = parse_generated_answer(ans.strip())
                N += 1
                output_records.append({
                    "qid": qids[i],
                    "query": queries[i],
                    "retrieved": rerank_infos[i],
                    "generated": pred,
                    "gold_answer": gold_answers[i],
                    "faiss_distances": D[i].tolist(),
                    "faiss_rowids": I[i].tolist(),
                    "gold_pids": list(gold_sets[i])
                })

    del llm
    del retriever
    del reranker
    gc.collect()
    torch.cuda.empty_cache()

    # =========================
    # Bi-Encoder eval（保持一致）
    # =========================
    sentence_scorer = SentenceTransformer(
        # "sentence-transformers/all-MiniLM-L6-v2",
        "/root/autodl-tmp/RAG-System-Model-Training/all-MiniLM-L6-v2",
        device=DEVICE
    )

    res = [r["generated"] for r in output_records]
    ans = [r["gold_answer"] for r in output_records]

    emb_res = sentence_scorer.encode(res, convert_to_tensor=True, normalize_embeddings=True)
    emb_gold = sentence_scorer.encode(ans, convert_to_tensor=True, normalize_embeddings=True)

    cos_scores = util.cos_sim(emb_res, emb_gold).diag().mean().item()

    print(f"Queries evaluated: {N}")
    print(f"Recall@{TOP_K}: {R_sum / max(N,1):.4f}")
    print(f"MRR@{TOP_K}: {MRR_sum / max(N,1):.4f}")
    print(f"Bi-Encoder CosSim: {cos_scores:.4f}")

    final = {"data_size": N,
             f"recall@{TOP_K}": R_sum / max(N, 1),
             f"mrr@{TOP_K}": MRR_sum / max(N, 1),
             "Bi-Encoder_CosSim": cos_scores,
             "records": output_records}

    os.makedirs("results", exist_ok=True)
    # result_file_path = os.path.join("results", args.result_file_name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file_path = os.path.join("results", f"{args.result_file_name}_{timestamp}.json")
    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)


# =========================
# 4. entry
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_folder", type=str, default="./vector_database")
    parser.add_argument("--index_file", type=str, default="passage_index.faiss")
    parser.add_argument("--sqlite_file", type=str, default="passage_store.db")
    parser.add_argument("--test_data_path", type=str, default="./data/test_open.txt")
    parser.add_argument("--qrels_path", type=str, default="./data/qrels.txt")
    parser.add_argument("--retriever_model_path", type=str, required=True)
    parser.add_argument("--reranker_model_path", type=str, required=True)
    parser.add_argument("--result_file_name", type=str, default="result.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_rag(args)
