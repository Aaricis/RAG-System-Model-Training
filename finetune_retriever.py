import argparse
import json
import logging
import os
from datetime import datetime

from datasets import Dataset
from sentence_transformers import InputExample, LoggingHandler
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

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
parser.add_argument("--max_seq_length", type=int, default=512)
args = parser.parse_args()

train_data_path = args.train_data_path
test_data_path = args.test_data_path
qrels_path = args.qrels_path
corpus_data_path = args.corpus_data_path
model_name = args.model_name

model_save_path = os.path.join(args.model_save_path,
                               f'train_bi-encoder-mnrl-{model_name.replace("/", "-")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
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

logger.info("构建训练数据。")


# 构造训练数据

queries, passages = [], []
with open(train_data_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        query = obj.get("rewrite")  # 锚点
        evidences = obj.get("evidences")
        retrieval_labels = obj.get("retrieval_labels")

        queries.append(query)
        for evidence, label in zip(evidences, retrieval_labels):
            if label == 1:
                passages.append(evidence)


train_dataset =  Dataset.from_dict({
    "anchor":   ["query: " + q for q in queries],
    "positive": ["passage: " + p for p in passages],
})

logger.info(f"成功建立 {len(train_dataset)} 个MNR训练样本。")

# 构造测试数据
# 查询 / anchor，relevant_docs（相关性标注），corpus（候选文档库）

logger.info("构建测试数据！")
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

logger.info(f"载入{len(corpus)}个段落，{len(queries)}个查询，{len(relevant_docs)}个相关性映射。")

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
model.max_seq_length = args.max_seq_length

# 4. 损失函数
loss = MultipleNegativesRankingLoss(model)

# 5. 训练

training_args = SentenceTransformerTrainingArguments(
    output_dir=model_save_path,

    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=batch_size,

    eval_strategy="steps",
    eval_steps=args.evaluation_steps,

    save_strategy="steps",
    save_steps=args.evaluation_steps,
    save_total_limit=2,

    logging_steps=50,
    report_to=["tensorboard"],

    fp16=True,
    warmup_ratio=0.1,

    load_best_model_at_end=True,
    metric_for_best_model="eval_test_recall_cosine_mrr@10",  # ✅ 修正
    greater_is_better=True,

    remove_unused_columns=False,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    evaluator=ir_evaluator,
    loss=loss,
)

logger.info("开始使用 SentenceTransformerTrainer 训练！")
trainer.train()
logger.info("训练完成！")
