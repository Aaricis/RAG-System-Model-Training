import argparse
import json
import logging
import os
import sys
from typing import Union, List, Dict, Counter

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util


def load_data(file_path: str, is_single_object: bool = False) -> Union[List[Dict], Dict]:
    """
    Loads data from a file. Handles JSONL (default) or a single large JSON object (e.g., qrels).
    """
    if not os.path.exists(file_path):
        print(
            f"Error: Required file not found at: {file_path}", file=sys.stderr)
        raise FileNotFoundError(f"Required file not found: {file_path}")

    if is_single_object:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise TypeError(
                        f"Expected a single JSON object (dict) in {file_path}, but got {type(data)}.")
                return data
            except json.JSONDecodeError as e:
                print(
                    f"FATAL JSON Error in single object file {file_path}: {e}", file=sys.stderr)
                return {}
    else:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(
                        f"Skipping malformed JSON line in {file_path}: Line {i + 1}. Error: {e}", file=sys.stderr)

        return data


def analyze_similarity(args):
    logging.info("Loading fine-tuned retriever model...")
    retriever = SentenceTransformer(args.retriever_model_path, device="cuda")

    logging.info(f"Loading dataset from {args.data_dir}")
    data = load_data(args.data_dir, is_single_object=False)

    qid_sim = {}
    for item in data:
        qid = item.get('qid')
        query_text_raw = item.get('rewrite', '').strip()
        # Apply E5 prefix to query
        query = f"query: {query_text_raw}"

        evidences = item.get('evidences', [])
        labels = item.get('retrieval_labels', [])
        positive = ''
        negatives = []
        for evidence, label in zip(evidences, labels):
            evidence_text = f"passage: {evidence.strip()}"
            if label == 1:
                positive = evidence_text
            else:
                negatives.append(evidence_text)

        gaps = cal_similarity(retriever, query, positive, negatives)
        qid_sim[qid] = gaps
    return qid_sim


def analyze_negative_difficulty(sims):
    stats = Counter()
    total_negs = 0

    for gaps in sims.values():
        total_negs += len(gaps)
        for gap in gaps:
            if gap > 0.30:
                stats["easy"] += 1
            elif gap > 0.10:
                stats["medium"] += 1
            elif gap > 0.0:
                stats["hard"] += 1
            else:
                stats["very_hard"] += 1

    logging.info("Negative difficulty distribution:\n")
    for k in ["easy", "medium", "hard", "very_hard"]:
        cnt = stats.get(k, 0)
        ratio = cnt / total_negs * 100
        logging.info(f"{k:10s}: {cnt:6d} ({ratio:6.2f}%)")

    return stats


def plot_stats(stats):
    """
    画分布直方图
    :param stats: stats
    :return:
    """

    # 数量分布直方图
    labels = ["easy", "medium", "hard", "very_hard"]
    counts = [stats.get(k, 0) for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.xlabel("Negative difficulty")
    plt.ylabel("Count")
    plt.title("Negative Difficulty Distribution (Count)")
    plt.tight_layout()

    # 保存
    save_dir = os.path.join(args.output_dir, "negative_difficulty_count.png")
    plt.savefig(save_dir, dpi=300)
    plt.close()

    # 占比分布直方图
    total = sum(stats.values())
    ratios = [stats.get(k, 0) / total * 100 for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, ratios)
    plt.xlabel("Negative difficulty")
    plt.ylabel("Percentage (%)")
    plt.title("Negative Difficulty Distribution (%)")

    # 在柱子上标数值
    for i, v in enumerate(ratios):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    plt.ylim(0, max(ratios) * 1.2)
    plt.tight_layout()
    # 保存
    save_dir = os.path.join(args.output_dir, "negative_difficulty_ratio.png")
    plt.savefig(save_dir, dpi=300)
    plt.close()


def cal_similarity(model, query, positive, negatives):
    """
    计算相似度gap
    :param model: retriever模型
    :param query: query
    :param positive: 正例
    :param negatives: 负例
    :return: gap
    """
    embs = model.encode(
        [query, positive] + negatives,
        normalize_embeddings=True,
    )

    q_emb = embs[0]
    p_emb = embs[1]
    n_embs = embs[2:]

    s_pos = util.cos_sim(q_emb, p_emb).item()
    s_negs = util.cos_sim(q_emb, n_embs).tolist()

    gaps = [s_pos - s for s in s_negs]
    return gaps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analysis the similarity between positive and negative samples")

    parser.add_argument("--data_dir", type=str, default="./data/train.txt",
                        help="Path to data file that will be analyzed")
    parser.add_argument("--retriever_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/similarity",
                        help="Directory to save the results")

    args = parser.parse_args()

    similarities = analyze_similarity(args)
    statistics = analyze_negative_difficulty(similarities)
    plot_stats(statistics)
