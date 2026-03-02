# RAG System Model Training

## RAG Pipeline with Fine-Tuned Retriever and Reranker
This project implements a complete, high-performance Retrieval-Augmented Generation (RAG) pipeline. It uses a "Retrieve-then-Rerank" two-stage process to find relevant context, which is then fed to a generative language model to produce an answer.

All components (Retriever, Reranker, and Generator prompts) have been fine-tuned and optimized on a custom Q&A dataset to achieve high scores across retrieval and generation metrics.
## 数据集简介
本数据集用于微调RAG系统retriever、reranker模型，以及离线训练强化学习策略。

数据集主要提供构建RAG系统所需的query、passage raw data，以及它们的对应关系，分为训练集和测试集供微调模型使用。

另外还包括自己采集的hard negative数据，用于reranker模型优化。以及自行构建的强化学习离线训练数据集。

## 数据格式与规范
- Corpus: passages to be retrieved

  ```json
  {
      "text": "...",
      "title": "...",
      "aid": "25749059",
      "bid": 5,
      "id": "25749059@5"
  }
  ```

- qrels: mappings of queries and their positive passages

  ```json
  {
      "qid1":{
          "passageId": 1
      }
  }
  ```

- Each query has a specific positive passage

- train / test_open: train and public test data

  ```json
  {
      "qid": "...",
      "rewrite": "...",
      "evidences": [],
       "answer": {
           "text": "",
           "answer_start": 0
       },
      "retrieval_labels": [0, 0, 0, 0, 1]
  }
  ```

   - 	rewrite: query content;
   - 	evidences: passages from BM25 negative sampling;
   - 	retrieval_labels: corresponding true/false label for passages in "evidences";
  
- The answer can be <u>an exact span of positive passage</u> or <u>CANNOTANSWER</u> in both train and test data;

- There can be no positive passage in "evidences" column in `test_open.txt`, however, you can find the positive passage id in `qrels.txt`.


- reranker_train_hard_neg_*.jsonl: hard negative mining

```json
{
  "query": "Where is Malayali located?",
  "passage": "According to the Indian census of 2001, there were 30,803,747 speakers of Malayalam in Kerala, making up 93.2% of the total number of Malayalam speakers in India, and 96.7% of the total population of the state. There were a further 701,673 (2.1% of the total number) in Karnataka, 557,705 (1.7%) in Tamil Nadu and 406,358 (1.2%) in Maharashtra. The number of Malayalam speakers in Lakshadweep is 51,100, which is only 0.15% of the total number, but is as much as about 84% of the population of Lakshadweep. In all, Malayalis made up 3.22% of the total Indian population in 2001. Of the total 33,066,392 Malayalam speakers in India in 2001, 33,015,420 spoke the standard dialects, 19,643 spoke the Yerava dialect and 31,329 spoke non-standard regional variations like Eranadan. As per the 1991 census data, 28.85% of all Malayalam speakers in India spoke a second language and 19.64% of the total knew three or more languages. Large numbers of Malayalis have settled in Bangalore, Mangalore, Delhi, Coimbatore, Hyderabad, Mumbai (Bombay), Ahmedabad, Pune, and Chennai (Madras). A large number of Malayalis have also emigrated to the Middle East, the United States, and Europe. Accessed November 22, 2014.</ref> including a large number of professionals. There were 7,093 Malayalam speakers in Australia in 2006. The 2001 Canadian census reported 7,070 people who listed Malayalam as their mother tongue, mostly in the Greater Toronto Area and Southern Ontario.",
  "label": 1
}
```

- offline_rl_query_3000.jsonl: reinforcement learning offline training data
```json
{
  "state": [
    1.4911106824874878,
    0.9573036432266235,
    0.7595851421356201,
    0.3282075524330139,
    0.2760377526283264,
    -0.06384341418743134,
    -0.4320544898509979,
    -0.4342767298221588,
    -0.5559507608413696,
    -2.326120138168335,
    0.4882858395576477,
    0.9990334510803223,
    0.8651432991027832,
    0.078125,
    1,
    -0.12435299903154373
  ],
  "action": 2,
  "reward": -0.5155330705185974,
  "next_state": [
    1.4911106824874878,
    0.9573036432266235,
    0.7595851421356201,
    0.3282075524330139,
    0.2760377526283264,
    -0.06384341418743134,
    -0.4320544898509979,
    -0.4342767298221588,
    -0.5559507608413696,
    -2.326120138168335,
    0.4882858395576477,
    0.9990334510803223,
    0.8651432991027832,
    0.078125,
    1,
    -0.12435299903154373
  ],
  "done": true,
  "qid": "C_191f92ba623c40e7b707183a363a4319_1_q#3",
  "top_m": 3,
  "reward_info": {
    "cosine": 0.7714184522628784,
    "rouge_l": 0.2755905511594023,
    "ctx_tokens": 1126,
    "quality": 0.6730546397149744,
    "cost": 0.886962068040758,
    "m_penalty": 0.03
  },
  "pred_ans": "Yes, the Eagles toured. They performed multiple tours, including the Hell Freezes Over Tour (1994-1996) and the Long Road Out of Eden Tour. They also reunited in 1994 and had subsequent tours without Leadon or Meisner."
}
```

## 数据集清单
