import argparse
import json
import logging
import os
from datetime import datetime

from datasets import Dataset
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainingArguments, CrossEncoderTrainer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
parser.add_argument("--test_data_path", type=str, default="./data/test_open.txt")
parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_save_path", type=str, default="./output/reranker")
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--evaluation_steps", type=int, default=500)
args = parser.parse_args()

short_model_name = args.model_name.split("/")[-1]
model_save_path = os.path.join(args.model_save_path,
                               f'train_cross-encoder-bcel-{short_model_name}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

os.makedirs(model_save_path, exist_ok=True)

# 设定日志
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# 加载训练数据
def load_train_data(file_path):
    """
    加载训练数据
    :param file_path: 文件路径
    :return: train dataset
    """
    queries, evidences, labels = [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = item["rewrite"]
            for evidence, label in zip(item["evidences"], item["retrieval_labels"]):
                queries.append(query)
                evidences.append(evidence)
                labels.append(float(label))

    data = {
        "sentence1": queries,
        "sentence2": evidences,
        "label": labels
    }

    pos_count = sum(1 for label in labels if label == 1.0)
    neg_count = len(labels) - pos_count
    logging.info(f"已加载 {len(labels)} 笔训练资料，正：{pos_count} / 负：{neg_count} (1:{neg_count / pos_count:.2f})")
    return Dataset.from_dict(data)


def load_test_data(file_path):
    """
    加载测试数据
    :param file_path: 文件路径
    :return: test dataset
    """
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if 1 not in item["retrieval_labels"]:  # 没有正例
                continue
            query = item["rewrite"]
            positives = []
            negatives = []
            for evidence, label in zip(item["evidences"], item["retrieval_labels"]):
                if label:
                    positives.append(evidence)
                else:
                    negatives.append(evidence)

            # 每个样本是一个三元组：查询，正例文档，负例文档列表
            samples.append(
                {
                    'query': query,
                    'positive': positives,
                    'negative': negatives
                }
            )
    logging.info(f'已加载 {len(samples)} 笔测试资料')
    return samples


# 1. 数据准备

## 训练集
train_dataset = load_train_data(args.train_data_path)

## 测试集
eval_samples = load_test_data(args.test_data_path)

# 2. 加载模型
model = CrossEncoder(model_name_or_path=args.model_name, num_labels=1, max_length=args.max_length)

# 3. 定义损失函数
loss = BinaryCrossEntropyLoss(model)

# 4. 评估器
reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=eval_samples,
    name="test_rerank",
    batch_size=args.batch_size,
    show_progress_bar=True
)

metrics = reranking_evaluator(model)
logging.info(f"Reranker模型微调前:{metrics}")


# 5. 定义训练参数

training_args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir=model_save_path,
    # Optional training parameters:
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    load_best_model_at_end=True,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=args.evaluation_steps,
    save_strategy="steps",
    save_steps=args.evaluation_steps,
    save_total_limit=2,
    logging_steps=50,
    logging_first_step=True,
    seed=12,

    report_to=["tensorboard"],
    # 关键修改：指定实际存在的指标
    metric_for_best_model="eval_test_rerank_mrr@10",  # ✅ 使用 MRR@10 作为评估指标

    # 重要：对于 MRR/MAP，越大越好，需设置 greater_is_better=True
    greater_is_better=True,
)

# 6. 创建Trainer并开启训练
trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    evaluator=reranking_evaluator,
    loss=loss
)

logging.info("开始使用CrossEncoderTrainer训练！")
trainer.train()
logger.info("训练完成！")
