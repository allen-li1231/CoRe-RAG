# Beyond Similarity: Complementary Retrieval for Efficient Multi-Hop Retrieval-Augmented Generation
> :bell: Notes
> Due to the limited file size allowed on Anonymous Github, we will open-source our model parameter, train and benchmark datasets after the review phrase.


## Table of Contents
- [Beyond Similarity: Complementary Retrieval for Efficient Multi-Hop Retrieval-Augmented Generation](#beyond-similarity-complementary-retrieval-for-efficient-multi-hop-retrieval-augmented-generation)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [System Requirement](#system-requirement)
    - [Python Environment](#python-environment)
    - [Embedding Databases](#embedding-databases)
  - [Multi-hop Retrieval with CoRe: How-to-Use](#multi-hop-retrieval-with-core-how-to-use)
  - [Paper Reproduction](#paper-reproduction)
  - [Training CoRe](#training-core)


## Introduction
CoRe (Complementary Retrieval) is a lightweight, embedding-level framework designed to address the computational inefficiencies of traditional recursive retrieval paradigm in the realm of Retrieval-Augmented Generation (RAG). CoRe eliminates the need for iterative LLM-based query rewriting while maintaining state-of-the-art performance. It achieves this through dynamic query embedding updates and complementary pruning strategies, enabling a streamlined "Retrieve-Embed-Retrieve" workflow.


## System Requirement
> Ubuntu 18.06 LTS+. \
> Nvidia GPU 48GB of RAM at minimum. \
> 16GB of system RAM for [reproduction](#paper-reproduction), 64GB for [training](#training-core). \
> 80GB of free space on hard drive.

### Python Environment
Please refer to [requirements.txt](/requirements.txt).

### Embedding Databases
We adopt [BGE-m3](https://arxiv.org/abs/2402.03216) as the default embedding model, upon which we also train our CoRe model for multi-hop retrieval.
Run the following two scripts to generate all necessary training and evaluate embedding databases.
You are **not required** to run them if you do not plan to train CoRe, as all necessary evaluate embedding databases are provided in the repository.

```sh
python init_hotpotqa_2wiki_musique_vectors.py
python init_multihop_rag_vectors.py
```


## Multi-hop Retrieval with CoRe: How-to-Use
Here we use [MultiHop RAG evaluate dataset](https://arxiv.org/abs/2401.15391) in the example below.
The repository comes with the necessary files for the example to run, see [preliminaries](#preliminaries).

```python
from embedding_rewriter.model import InfonceModel
from embedding_rewriter.hoyer import HoyerModel
from embedding_rewriter.asset_loader import get_embedding_rewriter_model
from passage_retrieval import MultiHopRetriever


EMBEDDING_NAME = "bge-m3"
EVALUATE_DATASET = "multihop_rag"

# load CoRe models from local checkpoints
infonce_model = get_embedding_rewriter_model(
    "checkpoint/infonce_...pt", InfonceModel, device="cuda"
)
hoyer_model = get_embedding_rewriter_model(
    "checkpoint/hoyer_...pt", HoyerModel, device="cuda"
)

# load retriever
retriever = MultiHopRetriever(
    "BAAI/bge-m3",
    passages=f"embedding_data/{EMBEDDING_NAME}/{EVALUATE_DATASET}/eval_passages.jsonl",
    passage_embeddings=f"embedding_data/{EMBEDDING_NAME}/{EVALUATE_DATASET}/eval_content_dense.npy",
    # uncomment this if faiss index is initialized, resulting in faster loading
    # faiss_index=f"embedding_data/{EMBEDDING_NAME}/{EVALUATE_DATASET}/",
    embedding_rewriter_model=[infonce_model, hoyer_model],
    projection_size=1024,
    save_or_load_index=True,
    indexing_batch_size=10240,
    index_device="cuda"     # or "cpu" on Apple Metal
)
```
> :bell: Notes
> * The passage jsonl file contains `id`, `title` and `text` for each passage in the retrieval database.
> * The `passage_embeddings` and `faiss_index` arguments accept a path to numpy embeddings or a faiss index directory, respectively. To replace them with your own database, please refer to logics in [init_hotpotqa_2wiki_musique_vectors.py](init_hotpotqa_2wiki_musique_vectors.py) and [init_multihop_rag_vectors.py](init_multihop_rag_vectors.py).
> * `embedding_rewriter_model` accepts a single model or a list of models. When two models (one `InfonceModel` and one `HoyerModel`) are provided, CoRe uses both as complementary retrievers.

The `retriever` has a `multihop_search_passages` method that supports retrieving both single queries and batch queries.

For single query:
```python
retrieve_result = retriever.multihop_search_passages(
    "Did Engadget report a discount on the 13.6-inch MacBook Air "
    "before The Verge reported a discount on Samsung Galaxy Buds 2?",
    n_hop=2,
    top_n=5
)
```

For batch queries:
```python
LIST_OF_QUESTIONS = [
    "Did Engadget report a discount on the 13.6-inch MacBook Air "
    "before The Verge reported a discount on Samsung Galaxy Buds 2?",
    "Did 'The Independent - Travel' report on Tremblant Ski Resort "
    "before 'Essentially Sports' mentioned Jeff Shiffrin's skiing habits?"
]

retrieve_result = retriever.multihop_search_passages(
    LIST_OF_QUESTIONS,
    n_hop=2,
    top_n=5,
    # change batch sizes on your device to optimize performance
    index_batch_size=2048,
    generate_batch_size=1024
)
```

To access retrieved passages and corresponding multi-hop retrieval paths:
```python
# retrieved passages for questions, organized by hop
# retrieve_result.passage is a list of length n_hop;
# each element is a list of length num_queries containing the passages retrieved at that hop.
print(retrieve_result.passage)

# employ networkx graph to depict multi-hop retrieval paths
retrieve_result = retriever.multihop_search_passages(
    LIST_OF_QUESTIONS,
    n_hop=2,
    top_n=5,
    index_batch_size=2048,
    generate_batch_size=1024,
    return_tree=True        # simply add this argument
)
# `retrieve_result.embedding_rewriter_graph` is a list of EmbeddingRewriterGraph objects
# (subclass of networkx.DiGraph) corresponding to the retrieval paths of the queries.
# Take the first query for example, to draw its retrieval path:
retrieval_graph = retrieve_result.embedding_rewriter_graph[0]
retrieval_graph.plot_tree()

# Nodes represent passages in the retrieval graph and store metadata:
print(retrieval_graph.nodes(data=True))
```


## Paper Reproduction
To evaluate the multi-hop retrieval performance of CoRe, run the following command. Here we take the 2WikiMultihop dataset and recall@5, tau=1, threshold=0.6 under three hops as an example.
The script will print recall rate and average number of retrieved passages at each hop, as well as statistics by question type.

> :bell: Notes
> * To change the evaluate dataset, replace `2wiki` with `hotpotqa_distractor`, `musique`, or `multihop_rag`.
> * Revise `n_hop` and `top_n` to change the number of hops and top retrieval settings.
> * Use `--model_cls infonce hoyer` and provide both checkpoint paths to `--model_name_or_path` to run the full CoRe system (complementary dual-retriever); use only one model for ablation.

```sh
python evaluation.py \
    --dataset_name 2wiki \
    --n_hop 3 \
    --top_n 5 \
    --model_cls infonce hoyer \
    --model_name_or_path "checkpoint/infonce_hotpotqa_distractor=0.360&musique=0.045&multihop_rag=0.080__epoch=16&loss=infonce&loss_alpha=0.1&x_size=1024&g_size=2048&mlp_size=2048&n_mlp=3&n_head=1&n_neg=10&neg_mode=paired&seed=1307.pt" "checkpoint/hoyer_hotpotqa_distractor=0.029&musique=0.039&multihop_rag=0.083__epoch=16&loss=hoyer&loss_alpha=0.1&x_size=1024&g_size=2048&mlp_size=2048&n_mlp=3&n_head=1&n_neg=10&neg_mode=paired&seed=1307.pt" \
    --threshold 0.6 \
    --tau 1 \
    --prune_redundant 
```


## Training CoRe
Run the following steps to generate embeddings and train the CoRe models. Please refer to the `parse_args` function in [training.py](./training.py) for a full description of training arguments.

**Step 1: Generate training embeddings**:
```sh
python init_hotpotqa_2wiki_musique_vectors.py
```

**Step 2: Train the InfoNCE model**:
```sh
python training.py \
    --graph_cache_dir ./train_data/ \
    --model_cls infonce \
    --loss infonce \
    --x_size 1024 \
    --g_size 2048 \
    --mlp_size 2048 \
    --n_mlp 3 \
    --n_head 1 \
    --lr 1e-4 \
    --n_neg 5 \
    --seed 1307
```

**Step 3: Training the Rewriters**:
To train the Similarity Rewriter:
```sh
python training.py \
    --graph_cache_dir ./train_data/ \
    --model_cls infonce \
    --loss infonce \
    --x_size 1024 \
    --g_size 2048 \
    --mlp_size 2048 \
    --n_mlp 3 \
    --n_head 1 \
    --lr 6e-5 \
    --n_neg 5 \
    --seed 1307
```
To train the Complementary Rewriter:
```sh
python training.py \
    --graph_cache_dir ./train_data/ \
    --model_cls hoyer \
    --loss hoyer \
    --x_size 1024 \
    --g_size 2048 \
    --mlp_size 2048 \
    --n_mlp 3 \
    --n_head 1 \
    --lr 1e-4s \
    --n_neg 5 \
    --seed 1307
```