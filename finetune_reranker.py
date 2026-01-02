import json
import logging
import argparse
from sentence_transformers import LoggingHandler
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
parser.add_argument("--test_data_path", type=str, default="./data/test_open.txt")
parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

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
            positive = ''
            negatives = []
            for evidence, label in zip(item["evidences"], item["retrieval_labels"]):
                if label:
                    positive = evidence
                else:
                    negatives.append(evidence)

            # 每个样本是一个三元组：查询，正例文档，负例文档列表
            samples.append((query, positive, negatives))
    logging.info(f'已加载 {len(samples)} 笔测试资料')
    return samples


# 1. 数据准备

# 训练集
train_dataset = load_train_data(args.train_data_path)

# 测试集
test_dataset = load_test_data(args.test_data_path)

# 2. 加载模型
model = CrossEncoder(model_name_or_path=args.model_name, num_labels=1, max_length=args.max_length)

# 3. 定义损失函数
loss = BinaryCrossEntropyLoss(model)

# 4. 评估器
reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=test_dataset,
    name="test_rerank",
    batch_size=args.batch_size,
    show_progress_bar=True
)

