import argparse
import os
import logging
import sys
import json
from typing import Union, List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3

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
    queries = {} # qid -> query_text
    qid_to_pos_pids = {} # qid -> set(pos_pid_1, pos_pid_2, ...)

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


