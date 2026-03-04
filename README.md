# RAG System Model Training

## RAG Pipeline with Fine-Tuned Retriever and Reranker
This project implements a complete, high-performance Retrieval-Augmented Generation (RAG) pipeline. It uses a "Retrieve-then-Rerank" two-stage process to find relevant context, which is then fed to a generative language model to produce an answer.

All components (Retriever, Reranker, and Generator prompts) have been fine-tuned and optimized on a custom Q&A dataset to achieve high scores across retrieval and generation metrics.

## Results

| 模型         | Recall@10 | MRR@10 (after rerank) | Bi-Encoder CosSim |
| ------------ | --------- | --------------------- | ----------------- |
| Base         | 0.8172    | 0.7240                | 0.3656            |
| Fine tune    | 0.8549    | 0.7737                | 0.3725            |
| RL-RAG-TOP-M | 0.8549    | 0.7737                | 0.3702            |

**详细实验过程和分析参见:**
- [我的博客](https://aaricis.github.io/posts/RAG-System-Model-Training/)

- [zhihu专栏](https://zhuanlan.zhihu.com/p/2011104267331724732)

## Quick Test
Download the pre-trained models and dataset from gdown and unzip them:
```shell
bash download.sh
```
Create `.env` at `./`:
```text
hf_token=YOUR_HUGGING_FACE_TOKEN
```
To quickly test the end-to-end pipeline with pre-trained models, run the following commands:
```shell
python save_embeddings.py --retriever_model_path ./models/retriever --build_db

python inference_batch.py --retriever_model_path ./models/retriever --reranker_model_path ./models/reranker --test_data_path ./data/test_open.txt
```
## Project Architecture
The pipeline consists of three main components:
1. Retriever(Bi-Encoder):
    - Model: `intfloat/multilingual-e5-small`
    - Task: Performing a fast, broad search over the entire corpus. It is fine-tuned on `(anchor, positive, negative_1, …, negative_n)` using `MultipleNegativesRankingLoss` to excel at finding relevant passages.
2. Reranker(Cross-Encoder):
    - Model: `cross-encoder/ms-marco-MiniLM-L12-v2`
    - Task: Taking the Top-K passages from the retriever and re-scores them with high precision. It is fine-tuned on a "hard-negative" dataset, where the negatives are passages that the retriever thought were quite good, forcing the reranker to learn fine-grained distinctions.
3. Generator(LLM):
    - Model: `Qwen/Qwen3-1.7B`
    - Task: Received the Top-M reranked passages and generates a final answer. Its performance is highly dependent on a "Natural Q&A" prompt provided in `utils.py`.

## Setup & Installation
1. Clone the repository and create a virtual environment with Python 3.12:
    ```shell
    git clone https://github.com/Aaricis/RAG-System-Model-Training.git
    cd RAG-System-Model-Training
    conda create -n <env_name> python=3.12
    conda activate <env_name>
    ```
2. Install dependencies:
    ```shell
    pip install -r requirements.txt
    ```
3. Hugging Face Login:
Create a `.env` file in the root directory and your Hugging Face token (required by `inference_batch.py`)
    ```text
    hf_token=YOUR_HUGGING_FACE_TOKEN
    ```
4. Dataset: 
Ensure your data files (train.txt, test_open.txt, corpus.txt, qrels.txt) are located in the `./data/` directory.

## Workflow & Usage
The "Retrieve-then-Rerank" pipeline must be trained in order.
- Step 1: Fine-Tune the Retriever

    This script trains the bi-encoder model on the `(anchor, positive, negative_1, …, negative_n)` found in `train.txt`.
    ```shell
    python finetune_retriever.py \
        --epochs 3 \
        --batch_size 32 \
        --grad_accumulate_step 4 \
        --learning_rate 2e-5 \
        --eval_batch_size 128 \
        --eval_steps 200
    ```
- Step 2: Build the Vector Database
    
    The `inference_batch.py` script requires a pre-built FAISS index and SQLite database. Using `save_embeddings.py` to create these using the fine-tuned retriever from Step 1 to encode `corpus.txt` and create the `passage_index.faiss` and `passage_store.db` in `./vector_database/.`.
    ```shell
    python3 save_embeddings.py \
        --retriever_model_path <retriever model path> \
        --build_db
    ```
- Step 3: Generate the Reranker Hard-Negative Dataset
    
    This script uses your fine-tuned retriever to mine hard negatives from the training set, creating a high-quality dataset to train the reranker.
    ```shell
    python3 generate_reranker_dataset.py \
        --retriever_model_path <retriever model path> \
        --output_file ./data/reranker_train_hard_neg.jsonl \
        --batch_size 64
    ```
- Step 4: Fine-Tune the Reranker
    
    This script trains the cross-encoder on the new hard-negative dataset.
    ```shell
    python3 finetune_reranker.py \
        --train_file ./data/reranker_train_hard_neg.jsonl \
        --epochs 1 \
        --batch_size 128 \
        --eval_batch_size 128 \
        --learning_rate 2e-5 \
        --eval_steps 200
    ```
- Step 5: Run Full RAG Inference & Evaluation
    
    This script runs the full, end-to-end pipeline (Retriever -> Reranker -> Generate) on the test set and saves the final results and metrics.
    ```shell
    python3 inference_batch.py \
        --retriever_model_path <retriever model path> \
        --reranker_model_path <reranker model path> \
        --test_data_path ./data/test_open.txt \
        --result_file_name "results.json"
    ```
## (Optional) Run RL-based Top-M tuning
Use **Reinforcement Learning** to train a model deciding the number of passages to include in the prompt.
- Step 1: Collect Dataset (once)

    Using your retriever and reranker and enumerates different reranker `top_m` to generate dataset.
    ```shell
    python rl_top-m.py \
	    --retriever_model_path <retriever model path> \
	    --reranker_model_path <reranker model path> \
	    --generator_model <generator model name or path> \
	    --scoring_model <scoring model name or path> \
	    --rl_offline_data ./data/offline_rl_query_3000.jsonl \
	    --mode collect \
	    --n_queries 3000
    ```
- Step 2: Offline PPO Train

    Using collected dataset from 'Step 1' to train PPO policy.
    ```shell
    python rl_top-m.py \
	    --rl_offline_data ./data/offline_rl_query_3000.jsonl \
	    --mode train
    ```
- Step 3: Online Inference

    Using PPO model that trained in 'Step 2' to infer `top_m` for every query during testing.
    ```shell
    python inference_batch_rl.py \
        --test_data_path ./data/test_open.txt \
        --retriever_model_path <retriever model path> \
        --reranker_model_path <reranker model path> \
        --rl_model_path ./output/rl.zip
    ```
## 免责声明 | Disclaimer

本项目仅供学习和研究使用。使用者须遵守当地的法律法规，包括但不限于 DMCA 相关法律。我们不对任何非法使用承担责任。

This project is for research and learning purposes only. Users must comply with local laws and regulations, including but not limited to DMCA-related laws. We do not take any responsibility for illegal usage.