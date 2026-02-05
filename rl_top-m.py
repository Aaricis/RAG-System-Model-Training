import gymnasium as gym
import numpy as np
from gymnasium import spaces
import argparse
import torch
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sqlite3
import faiss
import json


# 配置日志级别为INFO，输出到控制台
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- RAG Parameters ---
TOP_K = 10 # Retriever 檢索 K 篇
GEN_MAXLEN = 1280
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- RL Parameters ---
# Top_M 的最大值 (Action 空间是 1 到 TOP_K)
MAX_TOP_M = TOP_K

# 由于 llm 很慢，我们只用少量数据
N_TEST_CASES_FOR_RL = 100  # 只取前 n 笔 test data 来训练/测试 RL
TOTAL_TIMESTEPS = 100     # RL 总共只训练 n 步 (即跑 n 次 RAG 流程)


def load_test_data(test_path, qrels_path):
    """
    Load test data from file.
    :param test_path: test data path
    :param qrels_path: qrels path
    :return: tests
    """

    # Load qrels
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)
    qid2gold = {}
    for qid, pid2lab in qrels.items():
        gold = {pid for pid, lab in pid2lab.items() if str(lab) != "0"}
        qid2gold[qid] = gold

    # Load test queries
    tests = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            query = obj.get("rewrite")
            gold_answer = (obj.get("answer")).get("text", "")
            gold_pids = qid2gold.get(qid, set())

            # 必须要有黄金答案才能计算reward
            if query and gold_answer:
                tests.append({
                    "qid": qid,
                    "query": query,
                    "gold_answer": gold_answer,
                    "gold_pids": gold_pids,
                })
    return tests

# -----------------------
# RAG RL Environment
# -----------------------
class RAGEnv(gym.Env):
    """
    自定义gym环境，用于RAG流程中的Top_M选择。

    - Observation: Reranker排序后的Top_K个分数
    - Action: 选择Top_M (Discrete(K)，动作 0 对应 M=1，动作K-1对应 M=K)
    - Reward: 生成答案与黄金答案的 cosine similarity
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, args):
        super(RAGEnv, self).__init__()

        self.args = args
        self.top_k = TOP_K

        # --- 1. Define Action and Observation Spaces ---

        # Action: 选择 M=1 到 M=TOP_K
        # 动作0 -》M=1，动作1 -》M=2，...，动作9 -》M=10
        self.action_space = spaces.Discrete(TOP_K)

        # Observation:
        # [
        #   scores_norm(TopK),
        #   gap12_norm,
        #   gap1k_norm,
        #   entropy_norm,
        #   query_len_norm,
        #   avg_ctx_len_norm
        # ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.top_k + 5,), dtype=np.float32
        )

        logging.info("Initializing RAGEnv...")

        # --- 2. Load all models and data ---
        logging.info("Loading all models...")
        self.retriever = SentenceTransformer(args.retriever_model_path, device=DEVICE)
        self.reranker = CrossEncoder(args.reranker_model_path, device=DEVICE)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            args.generator_model,
            torch_dtype="auto",
            device_map = "auto"
        )
        self.scorer = SentenceTransformer(args.scoring_model, device=DEVICE)

        # --- 3. Load DB connection ---
        logging.info("Loading index and db...")
        sqlite_path = os.path.join(args.index_folder, args.sqlite_file)
        self.conn = sqlite3.connect(sqlite_path)
        self.cursor = self.conn.cursor()

        self.index = faiss.read_index(os.path.join(args.index_folder, args.index_file))

        # --- 4. Load Test Data ---
        self.tests = load_test_data(args.test_file, args.qrels_file)
        # 只使用一小部分数据
        if N_TEST_CASES_FOR_RL > 0:
            self.tests = self.tests[:N_TEST_CASES_FOR_RL]
        logging.info(f"Loaded {len(self.tests)} test cases...")
        self.current_test_index = 0

        # 存储当前episode数据
        self.current_reranked_data = []
        self.current_query = ""
        self.current_gold_answer = ""
        self.current_qid = ""

    def _get_obs_and_reranked_data(self):
        # ---- 1. Random sample test case ----
        idx = np.random.randint(0, len(self.tests))
        test_case = self.tests[idx]

        self.current_query = test_case["query"]
        self.current_gold_answer = test_case["gold_answer"]
        self.current_qid = test_case["qid"]

        # ---- 2. Retrieve ----
        prefix_query = "query: " + self.current_query
        q_emb = self.retriever.encode(
            [prefix_query], convert_to_numpy=True, normalize_embeddings=True
        )
        D, I = self.index.search(q_emb, self.top_k)

        rowids = I[0].tolist()
        need_rowids = set(int(rid) for rid in rowids)

        # ---- 3. DB fetch ----
        rowid2pt = {}
        if need_rowids:
            placeholders = ",".join(["?"] * len(need_rowids))
            sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
            rows = self.cursor.execute(sql, tuple(need_rowids)).fetchall()
            rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}

        cand_ids, cand_texts = [], []
        for rid in rowids:
            tup = rowid2pt.get(int(rid))
            if tup is None:
                continue
            cand_ids.append(tup[0])
            cand_texts.append(tup[1])

        # ---- 4. Rerank ----
        reranked_data = []
        scores_list = []

        if cand_texts:
            pairs = list(zip([self.current_query] * len(cand_texts), cand_texts))
            scores = self.reranker.predict(pairs, show_progress_bar=False)

            reranked_data = sorted(
                zip(scores, cand_ids, cand_texts),
                key=lambda x: x[0],
                reverse=True
            )
            scores_list = np.array([float(s) for s, _, _ in reranked_data], dtype=np.float32)

        self.current_reranked_data = reranked_data

        # ---- 5. Score padding ----
        if len(scores_list) == 0:
            scores_list = np.zeros(self.top_k, dtype=np.float32)

        if len(scores_list) < self.top_k:
            pad = np.ones(self.top_k - len(scores_list), dtype=np.float32) * scores_list.min()
            scores_list = np.concatenate([scores_list, pad])

        scores = scores_list[:self.top_k]

        # ---- 6. Normalize scores ----
        mu = scores.mean()
        std = scores.std() + 1e-6
        scores_norm = (scores - mu) / std

        # ---- 7. Gap + entropy ----
        gap12 = scores_norm[0] - scores_norm[1] if self.top_k > 1 else 0.0
        gap1k = scores_norm[0] - scores_norm[-1]

        p = np.exp(scores_norm)
        p = p / (p.sum() + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-8))
        entropy_norm = entropy / np.log(self.top_k + 1e-8)

        gap12_norm = np.tanh(gap12)
        gap1k_norm = np.tanh(gap1k)

        # ---- 8. Query token length (Chinese safe) ----
        q_tokens = self.llm_tokenizer.encode(self.current_query, add_special_tokens=False)
        q_len = len(q_tokens)
        MAX_Q_LEN = 64
        query_len_norm = min(q_len, MAX_Q_LEN) / MAX_Q_LEN

        # ---- 9. Context token cost ----
        ctx_lens = [
            len(self.llm_tokenizer.encode(t, add_special_tokens=False))
            for _, _, t in reranked_data[:self.top_k]
        ]

        if len(ctx_lens) == 0:
            avg_ctx_len_norm = 0.0
        else:
            avg_ctx = float(np.mean(ctx_lens))
            MAX_CTX_LEN = 256
            avg_ctx_len_norm = min(avg_ctx, MAX_CTX_LEN) / MAX_CTX_LEN

        # ---- 10. Final Observation ----
        obs = np.concatenate([
            scores_norm,
            [gap12_norm, gap1k_norm, entropy_norm, query_len_norm, avg_ctx_len_norm]
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs_and_reranked_data()
        info = {}
        return observation, info




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL model for RAG system.")

    # Data paths
    parser.add_argument("--train_file", type=str, default="./data/train.txt",
                        help="Path to the train.txt JSONL file (must contain evidences/retrieval_labels).")
    parser.add_argument("--qrels_file", type=str, default="./data/qrels.txt",
                        help="Path to the qrels.txt JSON file (single object).")
    parser.add_argument("--corpus_file", type=str, default="./data/corpus.txt",
                        help="Path to the corpus.txt JSONL file.")
    parser.add_argument("--test_file", type=str, default="./data/test_open.txt",
                        help="Path to the test_open.txt file for evaluation.")
    parser.add_argument("--output_dir", type=str, default="./output/rl",
                        help="Directory to save the trained model.")

    # Database paths
    parser.add_argument("--index_folder", type=str, default="./vector_database")
    parser.add_argument("--index_file", type=str, default="passage_index.faiss")
    parser.add_argument("--sqlite_file", type=str, default="passage_store.db")

    # Model paths
    parser.add_argument("--retriever_model_path", type=str, required=True,
                        help="Path to retriever model.")
    parser.add_argument("--reranker_model_path", type=str, required=True,
                        help="Path to the reranker model.")
    parser.add_argument("--generator_model", type=str, required=True,
                        help="Path to the generator model.")
    parser.add_argument("--scoring_model", type=str, required=True,
                        help="Path to the scoring model.")

    args = parser.parse_args()