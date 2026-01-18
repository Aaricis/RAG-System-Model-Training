import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Union, List, Dict

from datasets import Dataset
from sentence_transformers import LoggingHandler
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainingArguments, CrossEncoderTrainer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import mine_hard_negatives

# 设定日志
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


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


def prepare_train_examples(args):
    """
    Parses the pre-generated reranker training file (JSONL).
    """
    training_data = load_data(args.train_file, is_single_object=False)
    queries, evidences, labels = [], [], []
    skipped_count = 0

    # Create samples list
    for item in training_data:
        query = item.get('query')
        passage = item.get('passage')
        label = item.get('label')  # 1 for positive, 0 for negative

        if query and passage and isinstance(label, int):
            queries.append(query)
            evidences.append(passage)
            labels.append(label)
        else:
            skipped_count += 1

    # 生产环境健壮写法
    if len({len(queries), len(evidences), len(labels)}) != 1:
        raise ValueError("No valid training examples were generated. Training cannot proceed.")
    else:
        logging.info(f"Loaded {len(queries)} total triples. Skipped {skipped_count}.")

    train_examples = {
        "sentence1": queries,
        "sentence2": evidences,
        "label": labels
    }
    return Dataset.from_dict(train_examples)


def mine_negatives(args):
    """
    Add hard negatives to a dataset of (anchor, positive) pairs.
    """

    # 1. Load train dataset
    train_data = load_data(args.train_file, is_single_object=False)

    train_dataset = []
    for item in train_data:
        query = item.get('rewrite', '').strip()

        evidences = item.get('evidences', [])
        labels = item.get('retrieval_labels', [])

        pos = None
        for i, label in enumerate(labels):
            if label == 1:
                pos = evidences[i].strip()

        if pos:
            train_dataset.append({
                "query": f'query: {query}',  # Apply E5 prefix
                "positive": [f'passage: {pos}']  # Apply E5 prefix
            })

    # 2. Load Corpus (passages)
    corpus_data = load_data(args.corpus_file, is_single_object=False)
    corpus = []
    for item in corpus_data:
        text = item.get('text', '').strip()
        pid = item.get('id')
        if pid and text:
            corpus.append(f"passage: {text}")  # Apply E5 prefix

    # 3. Mining hard negatives
    model = SentenceTransformer(args.retriever_model_path, device='cuda')

    hard_dataset = mine_hard_negatives(
        model=model,
        dataset=Dataset.from_list(train_dataset),
        corpus=corpus,
        anchor_column_name="query",
        positive_column_name="positive",
        num_negatives=4,
        sampling_strategy="top",
        relative_margin=0.05,
        batch_size=args.batch_size,
        use_faiss=True,
        output_format="labeled-pair" # Apply for BinaryCrossEntropyLoss
    )

    return hard_dataset


def prepare_reranker_evaluator(args):
    """
    Loads test queries, corpus, qrels, and test_open.txt to create the
    CrossEncoderRerankingEvaluator.
    """
    logging.info("Loading data for Reranking Evaluator...")

    # 1. Load Corpus (passages)
    corpus_data = load_data(args.corpus_file, is_single_object=False)
    corpus_map = {}
    for item in corpus_data:
        text = item.get('text', '').strip()
        pid = item.get('id')
        if pid and text:
            corpus_map[pid] = text

    # 2. Load Qrels (ground truth relevant docs)
    qrels_data = load_data(args.qrels_file, is_single_object=True)

    # 3. Load Test Data (queries and negative passages)
    test_data = load_data(args.test_file, is_single_object=False)

    logging.info(f"Evaluator: Loaded {len(test_data)} queries, {len(corpus_map)} passages, {len(qrels_data)} qrels.")

    # Create samples list [ {'query': '...', 'positive': [list of texts], 'negative': [list of texts]} ]
    eval_samples = []
    skipped_queries = 0
    for item in test_data:
        qid = item.get("qid")
        query = item.get("rewrite", "").strip()

        if not qid or not query:
            skipped_queries += 1
            continue

        positive_texts = set()
        negative_texts = set()

        # Get the TRUE positive passages from qrels.txt
        true_positive_pids = qrels_data.get(qid, {}).keys()
        for pid in true_positive_pids:
            if pid in corpus_map:
                positive_texts.add(corpus_map[pid])

        # Get the NEGATIVE passages from test_open.txt's "evidences"
        passages = item.get('evidences', [])
        labels = item.get('retrieval_labels', [])

        for text, label in zip(passages, labels):
            if label == 0 and text.strip():
                negative_texts.add(text)

        # We must have at least one positive AND one negative to evaluate
        if positive_texts and negative_texts:
            eval_samples.append({
                'query': query,
                'positive': list(positive_texts),
                'negative': list(negative_texts)
            })
        else:
            skipped_queries += 1

    logging.info(f"Created {len(eval_samples)} valid samples for the reranking evaluator.")
    logging.info(f"Skipped {skipped_queries} queries (missing positives in qrels or negatives in evidences).")

    if not eval_samples:
        raise ValueError(
            "No valid evaluation samples were created. Check your test/qrels/corpus files.")

    return CrossEncoderRerankingEvaluator(
        samples=eval_samples,
        at_k=10,
        name="test_eval",
        batch_size=args.eval_batch_size,
        always_rerank_positives=False,
    )


def fine_tune_reranker(args):
    # 1. Load Model
    logging.info(f"Loading cross-encoder model: {args.model_name_or_path}")
    model = CrossEncoder(model_name_or_path=args.model_name_or_path, num_labels=1, max_length=512)

    # 2. Prepare Training Data
    train_examples = prepare_train_examples(args)

    # 3. Define the Loss Function
    train_loss = BinaryCrossEntropyLoss(model)

    # 4. Create the Evaluator
    evaluator = prepare_reranker_evaluator(args)

    metrics_before = evaluator(model)
    logging.info(f'Metrics Before: {metrics_before}')

    # 5. Train arguments
    output_dir = os.path.join(args.output_dir,
                              f'train_cross-encoder-bcel-{args.model_name_or_path.split("/")[-1]}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    os.makedirs(output_dir, exist_ok=True)

    training_args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
        load_best_model_at_end=True,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=50,
        logging_first_step=True,
        seed=12,

        report_to=["tensorboard"],
        # 关键修改：指定实际存在的指标
        metric_for_best_model="eval_test_eval_mrr@10",  # ✅ 使用 MRR@10 作为评估指标

        # 重要：对于 MRR/MAP，越大越好，需设置 greater_is_better=True
        greater_is_better=True,
    )

    # 6. Train the Model
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        loss=train_loss,
        evaluator=evaluator
    )

    logger.info("\nStarting fine-tuning...")
    trainer.train()
    logger.info(f"\nFine-tuning complete. Best model saved to {output_dir}")

    # Evaluate the final model
    metrics_after = evaluator(model)
    logging.info(f'Metrics After: {metrics_after}')

    # Save the final model
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a cross-encoder (reranker) model.")

    parser.add_argument("--model_name_or_path", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2",
                        help="Pretrained cross-encoder model to load.")
    # parser.add_argument("--retriever_model_path", type=str, required=True,
    #                     help="Retriever model to load.")

    # --- Data Path Arguments ---
    parser.add_argument("--output_dir", type=str, default="./output/reranker",
                        help="Directory to save the trained model.")
    parser.add_argument("--train_file", type=str, default="./data/reranker_train_hard_neg.jsonl",
                        help="Path to the pre-generated JSONL training file.")

    # --- Arguments for the evaluator ---
    parser.add_argument("--qrels_file", type=str, default="./data/qrels.txt",
                        help="Path to the qrels.txt JSON file (for evaluation).")
    parser.add_argument("--corpus_file", type=str, default="./data/corpus.txt",
                        help="Path to the corpus.txt JSONL file (for evaluation).")
    parser.add_argument("--test_file", type=str, default="./data/test_open.txt",
                        help="Path to the test_open.txt file (for evaluation).")

    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int,
                        default=1, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size. Cross-encoders require small batches (e.g., 8, 16, 32).")
    parser.add_argument("--eval_batch_size", type=int,
                        default=16, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Whether to use 16-bit precision (mixed precision).")
    parser.add_argument("--eval_steps", type=int,
                        default=100, help="Evaluation steps for evaluator")
    args = parser.parse_args()

    # hard_negatives = mine_negatives(args)

    fine_tune_reranker(args)