import argparse
import json
import logging
import os
import sqlite3
import sys
from typing import Union, List, Dict

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration ---
# How many passages to retrieve from the retriever to find hard negatives
TOP_K_RETRIEVER = 20


def load_data(file_path: str, is_single_object: bool = False) -> Union[List[Dict], Dict]:
    """Loads JSONL or a single JSON object from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")

    if is_single_object:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f"FATAL JSON Error in single object file {file_path}: {e}", file=sys.stderr)
                return {}
    else:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line {file_path}: Line {i + 1}", file=sys.stderr)
        return data


def create_hard_negatives(args):
    logging.info("Loading fine-tuned retriever model...")
    retriever = SentenceTransformer(args.retriever_model_path, device="cuda")

    logging.info("Loading FAISS index and SQLite DB...")
    index = faiss.read_index(os.path.join(args.index_folder, args.index_file))

    conn = sqlite3.connect(os.path.join(args.index_foler, args.sqlite_file))
    cur = conn.cursor()

    logging.info("Loading qrels (ground truth)...")
    # mapping of queries and their positive passages
    # {qid1: {passage_id1: 1}
    qrels_map = load_data(args.qrels_file, is_single_object=True)

    logging.info(f"Loading training queries from {args.train_file}...")
    train_queries_data = load_data(args.train_file, is_single_object=False)

    # Store queries and their true positive IDs
    queries = {}  # qid -> query_text
    qid_to_pos_pids = {}  # qid -> set(pos_pid_1, pos_pid_2, ...)

    for item in train_queries_data:
        qid = item.get("qid")
        query_text = item.get("rewrite", "").strip()
        if not qid or not query_text:
            continue
        queries[qid] = query_text

        # Get all positive PIDs for this query from qrels
        if qid in qrels_map:
            qid_to_pos_pids[qid] = {pid for pid in qrels_map[qid].key()}
        else:
            # Skip queries that have no positive passages in qrels
            continue

    query_list = list(queries.values())
    qid_list = list(queries.keys())
    logging.info(f"Found {len(query_list)} valid queries with ground truth in {args.train_file}.")

    # --- Start Mining ---
    new_training_examples = []

    for i in tqdm(range(0, len(query_list), args.batch_size), desc="Mining hard negatives"):
        batch_queries = query_list[i: i + args.batch_size]
        batch_qids = qid_list[i: i + args.batch_size]

        # 1. Retrieve top-k passages using the fine-tuned retriever
        prefix_queries = ["query: " + q for q in batch_queries]
        q_embs = retriever.encode(
            prefix_queries, convert_to_numpy=True, normalize_embeddings=True,
            batch_size=len(batch_queries), show_progress=False
        )

        # D = distances, I = rowids
        D, I = index.search(q_embs, TOP_K_RETRIEVER)

        # 2. Get all rowids to fetch from DB
        need_rowids = tuple(set(int(rid) for row in I for rid in row.tolist()))
        if not need_rowids:
            continue

        placeholders = ",".join(["?"] * len(need_rowids))
        sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
        rows = cur.execute(sql, need_rowids).fetchall()
        rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}

        # 3. Create (query, passage, label) pairs
        for j, qid in enumerate(batch_qids):
            query_text = queries[qid]
            true_pos_pids = qid_to_pos_pids[qid]

            # Get retrieved rowids for this query
            retrieved_rowids = I[j].tolist()

            # Store one true positive (if found)
            added_positive = False

            for rowid in retrieved_rowids:
                if rowid not in rowid2pt:
                    continue

                retrieved_pid, retrieved_text = rowid2pt[rowid]

                if retrieved_pid in true_pos_pids:
                    # This is a TRUE POSITIVE
                    new_training_examples.append(
                        {"query": query_text, "passage": retrieved_text, "label": 1}
                    )
                    added_positive = True
                else:
                    # This is a HARD NEGATIVE
                    new_training_examples.append(
                        {"query": query_text, "passage": retrieved_text, "label": 0}
                    )

            # If the retriever didn't find the positive, find it in the DB and add it
            if not added_positive:
                for pos_pid in true_pos_pids:
                    # Manually fetch this positive passage
                    sql = f"SELECT text FROM passages WHERE pid = ?"
                    res = cur.execute(sql, (pos_pid,)).fetchone()
                    if res:
                        pos_text = res[0]
                        new_training_examples.append(
                            {"query": query_text, "passage": pos_text, "label": 1}
                        )
    conn.close()

    # 4. Save the new dataset as JSONL
    logging.info(f"\nGenerated {len(new_training_examples)} new training pairs.")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for ex in new_training_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logging.info(f"New reranker training set saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hard negatives for reranker training.")

    parser.add_argument("--retriever_model_path", type=str, default="./models/retriever",
                        help="Path to the fine-tuned retriever model.")
    parser.add_argument("--index_folder", type=str, default="./vector_database")
    parser.add_argument("--index_file", type=str, default="passage_index.faiss")
    parser.add_argument("--sqlite_file", type=str, default="passage_store.db")

    parser.add_argument("--train_file", type=str, default="./data/train.txt",
                        help="Path to the original train.txt to get queries.")
    parser.add_argument("--qrels_file", type=str, default="./data/qrels.txt",
                        help="Path to the qrels.txt ground truth.")

    parser.add_argument("--output_file", type=str, default="./data/reranker_train_hard_neg.jsonl",
                        help="Path to save the new JSONL training file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for retriever encoding.")

    args = parser.parse_args()

    if not os.path.exists(args.retriever_model_path):
        logging.error(f"Retriever model not found at {args.retriever_model_path}")
        sys.exit(1)
