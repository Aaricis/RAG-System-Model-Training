import argparse
import json
import os
from datetime import datetime
import logging

from sentence_transformers import InputExample, losses, SentenceTransformer, LoggingHandler
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", type=str, default="./data/train.txt")
parser.add_argument("--test_data_path", type=str, default="./data/test_open.txt")
parser.add_argument("--qrels_path", type=str, default="./data/qrels.txt")
parser.add_argument("--corpus_data_path", type=str, default="./data/corpus.txt")
parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-small")
parser.add_argument("--model_save_path", type=str, default="./output/retriever")
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--evaluation_steps", type=int, default=500)
args = parser.parse_args()

train_data_path = args.train_data_path
test_data_path = args.test_data_path
qrels_path = args.qrels_path
corpus_data_path = args.corpus_data_path
model_name = args.model_name

model_save_path = args.model_save_path + f'train_bi-encoder-mnrl-{model_name.replace("/", "-")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(model_save_path, exist_ok=True)

batch_size = args.batch_size

# 设定日志
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# 1. 数据准备

def format_query(q):
    return f"query: {q}"


def format_passage(passage):
    return f"passage: {passage}"


# 构造训练数据

# anchor：查询/基准句（如用户的搜索问题）
# positive：与锚点语义相关的句子（如正确答案）
# negative：与锚点语义不相关的句子（如干扰项）

train_samples = []
with open(train_data_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        query = obj.get("rewrite")  # 锚点
        evidences = obj.get("evidences")
        retrieval_labels = obj.get("retrieval_labels")
        pos, neg = [], []  # 正例，负例
        for evidence, label in zip(evidences, retrieval_labels):
            if label == 1:
                pos.append(evidence)
            else:
                neg.append(evidence)

        # 三元组（锚点-正例-负例）
        for p in pos:
            for n in neg:
                # 正确格式：查询和文档都必须加前缀
                train_samples.append(
                    InputExample(texts=[format_query(query), format_passage(p), format_passage(n)])
                )
                # (query, positive_passage, negative_passage)

logger.info(f"成功建立 {len(train_samples)} 个MNR训练样本。")

# 构造测试数据
# 查询 / anchor，relevant_docs（相关性标注），corpus（候选文档库）
queries, relevant_docs, corpus = {}, {}, {}
"""
queries = {
    "q1": "如何申请信用卡？",
    "q2": "sentence transformer 如何训练？"
}

relevant_docs = {
    "q1": {"d1"},
    "q2": {"d2"}
}

corpus = {
    "d1": "信用卡申请流程说明",
    "d2": "使用 sentence-transformers 进行微调",
    "d3": "今天北京天气如何"
}
"""

with open(test_data_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        qid = obj.get("qid")
        query = obj.get("rewrite")
        retrieval_labels = obj.get("retrieval_labels")
        if 1 in retrieval_labels:
            queries[qid] = query

with open(qrels_path, "r", encoding="utf-8") as f:
    qrels = json.load(f)  # {qid: {passage1:1, passage2:1, ......}}

for qid in queries.keys():
    relevant_docs[qid] = []
    for pid, label in qrels[qid].items():
        if label == 1:  # 只需要正例
            relevant_docs[qid].append(pid)  # query_id → 一组相关 doc_id

texts = {}
with open(corpus_data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            p = json.loads(line)
            if p.get("id") and p.get("text"):
                texts[p["id"]] = p["text"]

for qid in queries.keys():
    for pid, _ in qrels[qid].items():
        corpus[pid] = texts[pid]

logger.info(f"载入 {len(corpus)} 个段落，{len(queries)} 个查询，{len(relevant_docs)} 个相关性映射。")


# 数据加载器
class SentenceDataset(Dataset):
    def __init__(self, samples: list[InputExample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


train_dataset = SentenceDataset(train_samples)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# 2. 评估器

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    show_progress_bar=True,
    mrr_at_k=[10],
    precision_recall_at_k=[1, 3, 5, 10],
    name="test_recall",
    batch_size=16,
)

# 3. 模型
model = SentenceTransformer(model_name)

# 4. 损失函数
loss_fn = losses.MultipleNegativesRankingLoss(model)

# 5. 训练

warmup_steps = (len(train_dataloader) // batch_size) * 0.1  # 总训练步数的10%

logger.info("开始模型训练！")
model.fit(
    train_objectives=[(train_dataloader, loss_fn)],
    evaluator=ir_evaluator,
    epochs=args.num_epochs,
    warmup_steps=warmup_steps,
    evaluation_steps=args.evaluation_steps,
    output_path=model_save_path,
    save_best_model=True,  # 儲存最佳模型
    use_amp=True,  # Automatic Mixed Precision (AMP)
)
logger.info("训练完成！")
