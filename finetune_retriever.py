import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Union, List, Dict

from transformers import AutoTokenizer

from datasets import Dataset
from sentence_transformers import LoggingHandler
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# parser = argparse.ArgumentParser()
# parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
# parser.add_argument("--test_data_path", type=str, default="./data/test_open.txt")
# parser.add_argument("--qrels_path", type=str, default="./data/qrels.txt")
# parser.add_argument("--corpus_data_path", type=str, default="./data/corpus.txt")
# parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-small")
# parser.add_argument("--model_save_path", type=str, default="./output/retriever")
# parser.add_argument("--num_epochs", type=int, default=1)
# parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--evaluation_steps", type=int, default=500)
# parser.add_argument("--max_seq_length", type=int, default=512)
# args = parser.parse_args()
#
# train_data_path = args.train_data_path
# test_data_path = args.test_data_path
# qrels_path = args.qrels_path
# corpus_data_path = args.corpus_data_path
# model_name = args.model_name
#
# model_save_path = os.path.join(args.model_save_path,
#                                f'train_bi-encoder-mnrl-{model_name.replace("/", "-")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
# os.makedirs(model_save_path, exist_ok=True)
#
# batch_size = args.batch_size
#
# 设定日志
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# # 1. 数据准备
#
# logger.info("构建训练数据。")
#
#
# # 构造训练数据
#
# queries, passages = [], []
# with open(train_data_path, "r", encoding="utf-8") as f:
#     for line in f:
#         if not line.strip():
#             continue
#         obj = json.loads(line)
#         query = obj.get("rewrite")  # 锚点
#         evidences = obj.get("evidences")
#         retrieval_labels = obj.get("retrieval_labels")
#
#         queries.append(query)
#         for evidence, label in zip(evidences, retrieval_labels):
#             if label == 1:
#                 passages.append(evidence)
#
#
# train_dataset =  Dataset.from_dict({
#     "anchor":   ["query: " + q for q in queries],
#     "positive": ["passage: " + p for p in passages],
# })
#
# logger.info(f"成功建立 {len(train_dataset)} 个MNR训练样本。")
#
# # 构造测试数据
# # 查询 / anchor，relevant_docs（相关性标注），corpus（候选文档库）
#
# logger.info("构建测试数据！")
# queries, relevant_docs, corpus = {}, {}, {}
# """
# queries = {
#     "q1": "如何申请信用卡？",
#     "q2": "sentence transformer 如何训练？"
# }
#
# relevant_docs = {
#     "q1": {"d1"},
#     "q2": {"d2"}
# }
#
# corpus = {
#     "d1": "信用卡申请流程说明",
#     "d2": "使用 sentence-transformers 进行微调",
#     "d3": "今天北京天气如何"
# }
# """
#
# with open(test_data_path, "r", encoding="utf-8") as f:
#     for line in f:
#         if not line.strip():
#             continue
#         obj = json.loads(line)
#         qid = obj.get("qid")
#         query = obj.get("rewrite")
#         retrieval_labels = obj.get("retrieval_labels")
#         if 1 in retrieval_labels:
#             queries[qid] = query
#
# with open(qrels_path, "r", encoding="utf-8") as f:
#     qrels = json.load(f)  # {qid: {passage1:1, passage2:1, ......}}
#
# for qid in queries.keys():
#     relevant_docs[qid] = []
#     for pid, label in qrels[qid].items():
#         if label == 1:  # 只需要正例
#             relevant_docs[qid].append(pid)  # query_id → 一组相关 doc_id
#
# texts = {}
# with open(corpus_data_path, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if line:
#             p = json.loads(line)
#             if p.get("id") and p.get("text"):
#                 texts[p["id"]] = p["text"]
#
# for qid in queries.keys():
#     for pid, _ in qrels[qid].items():
#         corpus[pid] = texts[pid]
#
# logger.info(f"载入{len(corpus)}个段落，{len(queries)}个查询，{len(relevant_docs)}个相关性映射。")
#
# # 2. 评估器
#
# ir_evaluator = InformationRetrievalEvaluator(
#     queries=queries,
#     corpus=corpus,
#     relevant_docs=relevant_docs,
#     show_progress_bar=True,
#     mrr_at_k=[10],
#     precision_recall_at_k=[1, 3, 5, 10],
#     name="test_recall",
#     batch_size=16,
# )
#
# # 3. 模型
# model = SentenceTransformer(model_name)
# model.max_seq_length = args.max_seq_length
#
# # 4. 损失函数
# loss = MultipleNegativesRankingLoss(model)
#
# # 5. 训练
#
# training_args = SentenceTransformerTrainingArguments(
#     output_dir=model_save_path,
#
#     num_train_epochs=args.num_epochs,
#     per_device_train_batch_size=batch_size,
#
#     eval_strategy="steps",
#     eval_steps=args.evaluation_steps,
#
#     save_strategy="steps",
#     save_steps=args.evaluation_steps,
#     save_total_limit=2,
#
#     logging_steps=50,
#     report_to=["tensorboard"],
#
#     fp16=True,
#     warmup_ratio=0.1,
#
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_test_recall_cosine_mrr@10",  # ✅ 修正
#     greater_is_better=True,
#
#     remove_unused_columns=False,
# )
#
# trainer = SentenceTransformerTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     evaluator=ir_evaluator,
#     loss=loss,
# )
#
# logger.info("开始使用 SentenceTransformerTrainer 训练！")
# trainer.train()
# logger.info("训练完成！")

# --- Configuration Constants ---
MAX_LENGTH = 512  # Max length for multilingual-e5-small

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
                        f"Skipping malformed JSON line in {file_path}: Line {i+1}. Error: {e}", file=sys.stderr)

        return data

def prepare_training_examples(args, model_name: str):
    """
        Parses data from train.txt, which contains the query, positive passage,
        and negative passages all in one object.

        This function does NOT load corpus.txt for training.
    """
    logging.info("Loading tokenizer and data files for pre-validation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_data = load_data(args.train_file, is_single_object=False)

    queries, passages = [], []
    skipped_count = 0
    logging.info(f"Creating (query, positive) pairs from {args.train_file}...")

    for item in train_data:
        query_text_raw = item.get('rewrite', '').strip()
        if not query_text_raw:
            skipped_count += 1
            continue

        # Apply E5 prefix to query
        query_text = f"query: {query_text_raw}"

        evidences = item.get('evidences', [])
        labels = item.get('retrieval_labels', [])

        if not evidences or not labels or len(evidences) != len(labels):
            skipped_count += 1
            continue

        positive_passage_text_raw = ""
        for i, label in enumerate(labels):
            if label == 1:
                passage_text = evidences[i].strip()
                if not passage_text:
                    continue
                positive_passage_text_raw = passage_text

        # If no positive passage, we can't create pairs.
        if not positive_passage_text_raw:
            skipped_count += 1
            continue

        # Apply E5 prefix to positive
        pos_passage_text = f"passage: {positive_passage_text_raw}"

        # --- Pre-validate query and positive passage ---
        query_token_len = len(tokenizer.encode(query_text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH))
        pos_token_len = len(tokenizer.encode(pos_passage_text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH))

        if not (query_token_len > 2 and pos_token_len > 2):
            # Skip if query or positive are invalid
            skipped_count += 1
            continue

        queries.append(query_text)
        passages.append(pos_passage_text)

    train_dataset =  Dataset.from_dict({
        "anchor":   queries,
        "positive": passages,
    })

    logging.info(f"Prepared {len(queries)} valid training triplets (from {len(train_data)} queries).")
    logging.info(f"Skipped {skipped_count} invalid items or triplet combinations.")

    return train_dataset


def prepare_evaluator(args):
    """
        Loads test queries, corpus, and qrels to create the InformationRetrievalEvaluator.
    """
    logging.info("Loading data for evaluator...")



def fine_tune_e5_small(args):
    # 1. Load Model
    logging.info(f"Loading model: {args.model_name_or_path}")
    model = SentenceTransformer(args.model_name_or_path)

    # 2. Prepare Training Data (NOW CREATES PAIRS from train.txt)
    train_examples = prepare_training_examples(args, args.model_name_or_path)

    # 3. Define the Loss Function (MNRL handles triplets and in-batch negatives)
    train_loss = MultipleNegativesRankingLoss(model)

    # 4. Prepare the Evaluator



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune multilingual-e5-small dense retriever with hard negatives.")

    parser.add_argument("--model_name_or_path", type=str, default="intfloat/multilingual-e5-small")

    # Data paths
    parser.add_argument("--output_dir", type=str, default="./models/retriever",
                        help="Directory to save the trained model.")
    parser.add_argument("--train_file", type=str, default="./data/train.txt",
                        help="Path to the train.txt JSONL file (must contain evidences/retrieval_labels).")
    parser.add_argument("--qrels_file", type=str, default="./data/qrels.txt",
                        help="Path to the qrels.txt JSON file (single object).")
    parser.add_argument("--corpus_file", type=str, default="./data/corpus.txt",
                        help="Path to the corpus.txt JSONL file.")
    parser.add_argument("--test_file", type=str, default="./data/test_open.txt",
                        help="Path to the test_open.txt file for evaluation.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=3,
                        help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per device (GPU).")
    parser.add_argument("--eval_batch_size", type=int,
                        default=16, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-5, help="The initial learning rate.")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Whether to use 16-bit precision (mixed precision).")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
