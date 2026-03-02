import faiss, json, torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import pickle
from tqdm import tqdm
import argparse
import sqlite3

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_folder", type=str, default="./data")
argparser.add_argument("--file_name", type=str, default="corpus.txt")
argparser.add_argument("--output_folder", type=str, default="./vector_database")
argparser.add_argument("--batch_size", type=int, default=256)
argparser.add_argument("--retriever_model_path", type=str, default="intfloat/multilingual-e5-small")
argparser.add_argument("--output_index_file_name", type=str, default="passage_index.faiss")
argparser.add_argument("--output_db_file_name", type=str, default="passage_store.db")
argparser.add_argument("--build_db", action="store_true", help="Whether to build the SQLite DB")
args = argparser.parse_args()

data_folder = args.data_folder
file_name = args.file_name
output_folder = args.output_folder
batch_size = args.batch_size
retriever_model_path = args.retriever_model_path
output_index_file_name = args.output_index_file_name
output_db_file_name = args.output_db_file_name

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(output_folder, exist_ok=True)

if args.build_db:
    sqlite_path = os.path.join(output_folder, output_db_file_name)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS passages")
    cur.execute("CREATE TABLE passages (rowid INTEGER PRIMARY KEY, pid TEXT, text TEXT)")
    with open(f"{data_folder}/{file_name}") as f:
        for idx, line in enumerate(tqdm(f, desc="buidling sqlite db")):
            obj = json.loads(line)
            cur.execute("INSERT INTO passages (rowid, pid, text) VALUES (?, ?, ?)", (idx, obj["id"], obj["text"]))
    conn.commit()
    conn.close()


retriever = SentenceTransformer(retriever_model_path, device=DEVICE)
passages_batch = []
index = None
with open(f"{data_folder}/{file_name}") as f:
    for idx, line in enumerate(tqdm(f, desc="building faiss index")):
        obj = json.loads(line)
        pid = obj["id"]
        text = "passage: " + obj["text"]  # add prefix if needed (multilingual-e5 recommend adding "passage: ")
        passages_batch.append(text)

        if len(passages_batch) >= batch_size:
            emb_mat = retriever.encode(passages_batch, convert_to_numpy=True,
                                       batch_size=batch_size, normalize_embeddings=True)
            if index is None:
                index = faiss.IndexFlatIP(emb_mat.shape[1])
            index.add(emb_mat)
            passages_batch = []  # reset batch


if passages_batch:
    emb_mat = retriever.encode(passages_batch, convert_to_numpy=True,
                               batch_size=batch_size, normalize_embeddings=True)
    if index is None:
        index = faiss.IndexFlatIP(emb_mat.shape[1])
    index.add(emb_mat)

faiss.write_index(index, os.path.join(output_folder, output_index_file_name))
