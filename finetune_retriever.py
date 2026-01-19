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

# 设定日志
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

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
                        f"Skipping malformed JSON line in {file_path}: Line {i + 1}. Error: {e}", file=sys.stderr)

        return data


def prepare_training_examples(args, model_name: str):
    """
        Parses data from train.txt, which contains the query, positive passage.
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
        query_token_len = len(
            tokenizer.encode(query_text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH))
        pos_token_len = len(
            tokenizer.encode(pos_passage_text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH))

        if not (query_token_len > 2 and pos_token_len > 2):
            # Skip if query or positive are invalid
            skipped_count += 1
            continue

        queries.append(query_text)
        passages.append(pos_passage_text)

    train_dataset = Dataset.from_dict({
        "anchor": queries,
        "positive": passages,
    })

    logging.info(f"Prepared {len(queries)} valid training pairs (from {len(train_data)} queries).")
    logging.info(f"Skipped {skipped_count} invalid items or triplet combinations.")

    return train_dataset


def prepare_evaluator(args):
    """
        Loads test queries, corpus, and qrels to create the InformationRetrievalEvaluator.
    """
    logging.info("Loading data for evaluator...")

    # 1. Load Corpus (passages)
    corpus_data = load_data(args.corpus_file, is_single_object=False)
    corpus = {}
    for item in corpus_data:
        text = item.get('text', '').strip()
        pid = item.get('id')
        if pid and text:
            corpus[pid] = f"passage: {text}"  # Apply E5 prefix

    # 2. Load Test Queries
    test_data = load_data(args.test_file, is_single_object=False)
    queries = {}
    for item in test_data:
        qid = item.get('qid')
        query_text = item.get('rewrite', '').strip()
        if qid and query_text:
            queries[qid] = f"query: {query_text}"  # Apply E5 prefix

    # 3. Load Qrels (relevant docs)
    qrels_data = load_data(args.qrels_file, is_single_object=True)
    relevant_docs = {}
    for qid, pids_map in qrels_data.items():
        if qid in queries:
            relevant_docs[qid] = set(pids_map.keys())

    logging.info(f"Evaluator: Loaded {len(queries)} queries, {len(corpus)} passages, {len(relevant_docs)} qrels.")

    return InformationRetrievalEvaluator(
        queries,
        corpus,
        relevant_docs,
        mrr_at_k=[10, 20, 50],
        accuracy_at_k=[1, 5, 10, 20, 50],
        precision_recall_at_k=[1, 5, 10, 20, 50],
        name="test_eval",
        batch_size=args.eval_batch_size,
    )


def fine_tune_e5_small(args):
    # 1. Load Model
    logging.info(f"Loading model: {args.model_name_or_path}")
    model = SentenceTransformer(args.model_name_or_path)

    # 2. Prepare Training Data (NOW CREATES PAIRS from train.txt)
    train_examples = prepare_training_examples(args, args.model_name_or_path)

    # 3. Define the Loss Function (MNRL handles triplets and in-batch negatives)
    train_loss = MultipleNegativesRankingLoss(model, scale=30)

    # 4. Prepare the Evaluator
    evaluator = prepare_evaluator(args)

    metrics_before = evaluator(model)
    logging.info(f'Metrics Before: {metrics_before}')

    # 5. Train arguments

    output_dir = os.path.join(args.output_dir,
                              f'train_bi-encoder-mnrl-{args.model_name_or_path.replace("/", "-")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    os.makedirs(output_dir, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,

        logging_steps=50,
        report_to=["tensorboard"],

        fp16=True,
        warmup_ratio=0.1,
        learning_rate=args.learning_rate,

        load_best_model_at_end=True,
        metric_for_best_model="eval_test_eval_cosine_mrr@10",  # ✅ 修正
        greater_is_better=True,

        remove_unused_columns=False,
    )

    # 6.Train the Model
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        evaluator=evaluator,
        loss=train_loss,
    )

    logger.info("\nStarting fine-tuning...")
    trainer.train()
    logger.info(f"\nFine-tuning complete. Best model saved to {output_dir}")

    # 拿到best model
    best_model = trainer.model

    # 导出为SentenceTransformer模型
    best_model.save(output_dir)

    # 用best model做最终评估
    metrics_after = evaluator(best_model)
    logging.info(f'Metrics After: {metrics_after}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune multilingual-e5-small dense retriever with hard negatives.")

    parser.add_argument("--model_name_or_path", type=str, default="intfloat/multilingual-e5-small")

    # Data paths
    parser.add_argument("--output_dir", type=str, default="./output/retriever",
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
    parser.add_argument("--eval_steps", type=int,
                        default=100, help="Evaluation steps for evaluator")

    args = parser.parse_args()

    fine_tune_e5_small(args)
