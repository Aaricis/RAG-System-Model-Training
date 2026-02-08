import argparse
import gc
import json
import logging
import os
import random
import sqlite3
import sys

import faiss
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_inference_system_prompt, get_inference_user_prompt, parse_generated_answer
from collections import defaultdict

# 配置日志级别为INFO，输出到控制台
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- RAG Parameters ---
TOP_K = 10  # Retriever 檢索 K 篇
GEN_MAXLEN = 1280
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- RL Parameters ---
# Top_M 的最大值 (Action 空间是 1 到 TOP_K)
MAX_TOP_M = TOP_K

# 用少量数据测试
TRAIN_DATA_SIZE = 100  # 只取前 n 笔 test data 来训练/测试 RL


def load_training_data(data_path: str, qrels_path: str):
    """
    Load test data from file.
    :param data_path: test data path
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
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
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
                data.append({
                    "qid": qid,
                    "query": query,
                    "gold_answer": gold_answer,
                    "gold_pids": gold_pids,
                })
    return data


def load_offline_data(file_path: str):
    if not os.path.exists(file_path):
        print(
            f"Error: Required file not found at: {file_path}", file=sys.stderr)
        raise FileNotFoundError(f"Required file not found: {file_path}")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"Skipping malformed JSON line in {file_path}: Line {i + 1}. Error: {e}", file=sys.stderr)

    return data


# Rouge-L (Chinese Safe)
def lcs_length(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]


def rouge_l_f1(pred, gold):
    if len(pred) == 0 or len(gold) == 0:
        return 0.0

    pred_chars = list(pred)
    gold_chars = list(gold)

    lcs = lcs_length(pred_chars, gold_chars)

    prec = lcs / (len(pred_chars) + 1e-8)
    rec = lcs / (len(gold_chars) + 1e-8)

    if prec + rec == 0:
        return 0.0

    f1 = (2 * prec * rec) / (prec + rec)
    return float(f1)


# -----------------------
# RAG RL Environment
# -----------------------

def compute_reward(
        pred_ans: str,
        gold_ans: str,
        reranked_data,
        top_m: int,
        scorer,
        tokenizer,
        w_cos=0.6,
        w_rouge=0.4,
        lambda_cost=0.2,
):
    # -------- Semantic --------
    emb_res = scorer.encode([pred_ans], convert_to_tensor=True, normalize_embeddings=True)
    emb_gold = scorer.encode([gold_ans], convert_to_tensor=True, normalize_embeddings=True)
    cos = util.cos_sim(emb_res, emb_gold)[0][0].item()
    cos_norm = (cos + 1) / 2

    # -------- Lexical --------
    rouge = rouge_l_f1(pred_ans, gold_ans)

    # -------- Confidence --------
    conf = np.tanh(len(pred_ans) / 50)

    quality = (
        w_cos * cos_norm +
        w_rouge * rouge +
        0.1 * conf
    )
    quality = np.clip(quality, 0, 1)

    # -------- Token Cost --------
    ctx_tokens = sum(
        len(tokenizer.encode(t, add_special_tokens=False))
        for _, _, t in reranked_data[:top_m]
    )

    cost = np.tanh(ctx_tokens / 800)

    # -------- Marginal M Penalty --------
    m_penalty = 0.01 * top_m

    # -------- Final --------
    reward = quality - lambda_cost * cost - m_penalty

    # -------- PPO Safe --------
    reward = float(np.tanh(reward))

    return reward, {
        "cosine": cos_norm,
        "rouge_l": rouge,
        "ctx_tokens": ctx_tokens,
        "quality": quality,
        "cost": cost,
        "m_penalty": m_penalty
    }



class RAGEnv(gym.Env):
    """
    自定义gym环境，用于RAG流程中的Top_M选择。

    - Observation: Reranker排序后的Top_K个分数
    - Action: 选择Top_M (Discrete(K)，动作 0 对应 M=1，动作K-1对应 M=K)
    - Reward: 生成答案与黄金答案的 cosine similarity
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, args, mode="collect", offline_dataset=None):
        super(RAGEnv, self).__init__()

        self.args = args
        self.top_k = TOP_K
        self.mode = mode

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
            low=-5.0, high=5.0, shape=(self.top_k + 6,), dtype=np.float32
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
            device_map="auto"
        )
        self.scorer = SentenceTransformer(args.scoring_model, device=DEVICE)

        # --- 3. Load DB connection ---
        logging.info("Loading index and db...")
        sqlite_path = os.path.join(args.index_folder, args.sqlite_file)
        self.conn = sqlite3.connect(sqlite_path)
        self.cursor = self.conn.cursor()

        self.index = faiss.read_index(os.path.join(args.index_folder, args.index_file))

        # --- 4. Load Test Data ---
        if self.mode == "collect":
            self.train_data = load_training_data(args.train_file, args.qrels_file)
            # 只使用一小部分数据
            if TRAIN_DATA_SIZE > 0:
                self.train_data = self.train_data[:TRAIN_DATA_SIZE]
            logging.info(f"Loaded {len(self.train_data)} training samples...")
            self.current_test_index = 0

        if self.mode == "train":
            self.offline_dataset = offline_dataset
            random.shuffle(self.offline_dataset)
            self.offline_ptr = 0

        # 存储当前episode数据
        self.current_reranked_data = []
        self.current_query = ""
        self.current_gold_answer = ""
        self.current_qid = ""

    def _get_obs_and_reranked_data(self):
        # ---- 1. Random sample test case ----
        idx = np.random.randint(0, len(self.train_data))
        test_case = self.train_data[idx]

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
        scores_norm = np.clip(scores_norm, -3, 3)

        # ---- 7. Gap + entropy ----
        gap12 = scores_norm[0] - scores_norm[1] if self.top_k > 1 else 0.0
        gap1k = scores_norm[0] - scores_norm[-1]

        p = np.exp(scores_norm - scores_norm.max())
        p = p / (p.sum() + 1e-8)
        p = np.clip(p, 1e-8, None)

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

        if len(ctx_lens) > 2:
            slope = np.mean(np.diff(ctx_lens[:5]))
        else:
            slope = 0.0

        slope_norm = np.tanh(slope / 50)

        # ---- 10. Final Observation ----
        obs = np.concatenate([
            scores_norm,
            [gap12_norm, gap1k_norm, entropy_norm, query_len_norm, avg_ctx_len_norm, slope_norm]
        ]).astype(np.float32)

        return obs

    def evaluate_action(self, action):
        """
        Evaluate Top_M without changing episode state.
        Used only for offline dataset collection.
        """
        top_m = int(action) + 1

        if len(self.current_reranked_data) == 0:
            context_list = []
        else:
            context_list = [text for _, _, text in self.current_reranked_data[:top_m]]

        # ---- Generate ----
        try:
            messages = [
                {"role": "system", "content": get_inference_system_prompt()},
                {"role": "user", "content": get_inference_user_prompt(self.current_query, context_list)}
            ]

            self.llm_tokenizer.padding_side = "left"
            rendered_prompt = self.llm_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            inputs = self.llm_tokenizer(
                [rendered_prompt], padding=True, return_tensors="pt"
            ).to(self.llm_model.device)

            with torch.no_grad():
                outs = self.llm_model.generate(**inputs, max_new_tokens=GEN_MAXLEN)

            decoded = self.llm_tokenizer.batch_decode(outs, skip_special_tokens=True)
            pred_ans = parse_generated_answer(decoded[0].strip())
            if pred_ans.strip() == "":
                pred_ans = " "

        except Exception as e:
            print("LLM generation failed:", e)
            pred_ans = " "

        # ---- Reward ----
        reward, reward_info = compute_reward(
            pred_ans=pred_ans,
            gold_ans=self.current_gold_answer,
            reranked_data=self.current_reranked_data,
            top_m=top_m,
            scorer=self.scorer,
            tokenizer=self.llm_tokenizer,
            w_cos=0.6,
            w_rouge=0.4,
            lambda_cost=0.2
        )

        return reward, reward_info, pred_ans

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs_and_reranked_data()
        info = {}

        return observation, info

    def step(self, action):
        if self.mode == "train":
            sample = self.offline_dataset[self.offline_ptr]
            self.offline_ptr = (self.offline_ptr + 1) % len(self.offline_dataset)

            obs = np.array(sample["state"], dtype=np.float32)
            next_obs = np.array(sample["next_state"])

            reward = float(sample["reward"])

            terminated = True
            truncated = False

            return next_obs, reward, terminated, truncated, {}

    def close(self):
        logging.info("Closing RAGEnv......")
        self.conn.close()
        del self.retriever, self.reranker, self.llm_model, self.llm_tokenizer, self.scorer
        gc.collect()
        torch.cuda.empty_cache()


def collect_offline_dataset(env, save_path):
    dataset = []

    for _ in range(len(env.train_data)):
        # ---- reset: retrieve + rerank once ----
        obs, _ = env.reset()

        # cache obs
        state = obs.tolist()

        # ---- evaluate all Top_M on same retrieval ----
        for m in range(env.top_k):
            reward, reward_info, pred_ans = env.evaluate_action(m)

            dataset.append({
                "state": state,
                "action": int(m),
                "reward": float(reward),
                "next_state": state,
                "done": True,
                "qid": env.current_qid,
                "top_m": m + 1,
                "reward_info": reward_info,
                "pred_ans": pred_ans
            })

    logging.info(f"Generated {len(dataset)} offline samples.")

    # reward normalize per-query
    # PPO 非常吃 reward scale

    by_qid = defaultdict(list)
    for x in dataset:
        by_qid[x["qid"]].append(x)

    for qid, items in by_qid.items():
        rs = [i["reward"] for i in items]
        mu, std = np.mean(rs), np.std(rs) + 1e-6
        for i in items:
            i["reward"] = (i["reward"] - mu) / std

    with open(save_path, "w", encoding="utf-8") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    logging.info(f"Generated offline samples saved to {save_path}.")


def train(args):
    logging.info("Offline PPO for RAG Top-M")

    # -------- Phase 1: Collect --------
    if args.mode == "collect":
        logging.info("Collecting offline dataset...")
        env = RAGEnv(args, mode="collect")
        collect_offline_dataset(env, args.rl_offline_data)
        env.close()

    elif args.mode == "train":
        # -------- Phase 2: Train --------
        logging.info("Training PPO offline...")
        offline_dataset = load_offline_data(args.rl_offline_data)
        env = DummyVecEnv([lambda: RAGEnv(args, mode="offline", offline_dataset=offline_dataset)])

        # Contextual Bandit PPO
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=1.0,
            gae_lambda = 1.0,
            learning_rate=5e-4,
            n_steps=64,
            batch_size=64,
            ent_coef=0.05,
            clip_range=0.2,
            vf_coef=0.1
        )

        model.learn(total_timesteps=30000)
        model.save(args.output_dir)
        logging.info(f"RL model saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RL model for RAG system.")

    # Data paths
    parser.add_argument("--train_file", type=str, default="./data/train.txt",
                        help="Path to the train.txt JSONL file (must contain evidences/retrieval_labels).")
    parser.add_argument("--qrels_file", type=str, default="./data/qrels.txt",
                        help="Path to the qrels.txt JSON file (single object).")
    parser.add_argument("--output_dir", type=str, default="./output/rl",
                        help="Directory to save the trained model.")
    parser.add_argument("--rl_offline_data", type=str, default="./data/offline_rl.json")

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

    # Env mode
    parser.add_argument("--mode", type=str, choices=["collect", "train"], default="collect", required=True,
                        help="Online Collect or Offline Train")

    args = parser.parse_args()

    # 固定seed
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    train(args)
