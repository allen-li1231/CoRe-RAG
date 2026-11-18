import os
os.environ["HF_HUB_OFFLINE"] = '1'
os.environ["DGLBACKEND"] = "pytorch"
import argparse
import pickle
import functools
from collections.abc import Iterable
import pandas as pd

import src
from src.utils import DEVICE
from src.language_models import MODEL_DICT
from passage_retrieval import MultiHopRetriever
from embedding_rewriter.model import EmbeddingRewriterModel
from embedding_rewriter.asset_loader import get_dataset, get_embedding_rewriter_model


if DEVICE == "mps":
    DEVICE = "cpu"


@functools.lru_cache(maxsize=4)
def get_embedding_model(embedding_name: str, device: str):
    """Load and cache embedding models to avoid re-loading the same model multiple times.
    
    Args:
        embedding_name: Name of the embedding model (e.g., "bge-m3")
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if embedding_name not in MODEL_DICT:
        raise ValueError(f"Unknown embedding name: {embedding_name}, "
                        f"available options are: {list(MODEL_DICT.keys())}")
    
    embedding_model, embedding_tokenizer, _ = src.load_retriever(
        MODEL_DICT[embedding_name]
    )
    embedding_model.eval()
    embedding_model = embedding_model.to(device)
    
    return embedding_model, embedding_tokenizer


@functools.lru_cache()
def get_retriever(
    dataset_name: str,
    embedding_name: str,
    embedding_backend: str,
    model: EmbeddingRewriterModel | Iterable[EmbeddingRewriterModel],
    indexing_batch_size=1024,
    index_device=DEVICE,
    embedding_model=None,
    embedding_tokenizer=None,
    **backend_kwargs
):
    """Create MultiHopRetriever for given dataset and models
    
    Args:
        embedding_model: Pre-loaded embedding model (optional)
        embedding_tokenizer: Pre-loaded embedding tokenizer (optional)
    """
    if embedding_name not in MODEL_DICT:
        raise ValueError(f"Unknown embedding name: {embedding_name},"
                         f" available options are: {list(MODEL_DICT.keys())}")

    # check if is iterable of models, if not, convert to tuple
    if not isinstance(model, Iterable):
        model = (model,)

    x_size = next(iter(model)).x_size
    assert all(isinstance(m, EmbeddingRewriterModel) for m in model), \
        f"Models must be PyTorch modules, got {[type(m) for m in model]}"
    assert all(m.x_size == x_size for m in model), \
        f"All models must have the same x_size, got {[m.x_size for m in model]}"

    retriever = MultiHopRetriever(
        MODEL_DICT[embedding_name],
        passages=f"embedding_data/{embedding_name}/{dataset_name}/eval_passages.jsonl",
        passage_embeddings=f"embedding_data/{embedding_name}/{dataset_name}/eval_content_dense.npy",
        backend=embedding_backend,
        save_or_load_index=True,
        faiss_index=f"embedding_data/{embedding_name}/{dataset_name}/",
        projection_size=x_size,
        embedding_rewriter_model=model,
        indexing_batch_size=indexing_batch_size,
        index_device=index_device,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
        **backend_kwargs
    )
    return retriever


def evaluate_retrieve(
    model,
    n_hop,
    top_n,
    prune_redundant=True,
    prune_layer_top=True,
    threshold=0.,
    tau=2,
    index_batch_size=1024,
    generate_batch_size=1024,
    embedding_name="bge-m3",
    embedding_backend="hf"
):
    """Evaluate model across multiple datasets with cached embedding model loading.
    
    Args:
        model: The embedding rewriter model to evaluate
        n_hop: Number of hops for multihop retrieval
        top_n: Number of retrieved chunks for each hop
        index_batch_size: Batch size for Faiss retrieval
        generate_batch_size: Batch size for model inference
        embedding_name: Name of the embedding model to use
        embedding_backend: Backend for embedding retrieval (hf or vllm)
        prune_redundant: Whether to prune redundant passages
        prune_layer_top: Whether to prune top layers
        threshold: Similarity threshold for filtering
        tau: Tau value for multihop retrieval

        
    Returns:
        Dictionary mapping dataset names to their evaluation statistics
    """
    # Load embedding model using cached function to avoid re-loading the same model
    # The get_embedding_model function is cached, so the same model will be reused
    # across multiple calls to evaluate_retrieve
    embedding_model, embedding_tokenizer = get_embedding_model(embedding_name, DEVICE)

    datasets = ["hotpotqa_distractor", "musique", "multihop_rag"]
    d_stats = {}
    for dataset_name in datasets:
        stat = evaluate_dataset(
            model=model,
            dataset_name=dataset_name,
            embedding_name=embedding_name,
            embedding_backend=embedding_backend,
            n_hop=n_hop,
            top_n=top_n,
            prune_redundant=prune_redundant,
            prune_layer_top=prune_layer_top,
            threshold=threshold,
            tau=tau,
            index_batch_size=index_batch_size,
            generate_batch_size=generate_batch_size,
            embedding_model=embedding_model,
            embedding_tokenizer=embedding_tokenizer
        )
        d_stats[dataset_name] = stat

    return d_stats


def match_retrieve(df, retrieved_passages):
    set_title = df["set_evidence_title"].copy()
    idx_result = df.name

    lst_match = [0] * len(retrieved_passages)
    for i_hop, retrieved in enumerate(retrieved_passages):
        passage = retrieved[idx_result]
        for psg in passage:
            if psg["title"] not in set_title:
                continue

            set_title.remove(psg["title"])
            lst_match[i_hop] += 1

    return lst_match


def evaluate_dataset(
    model,
    dataset_name,
    embedding_name,
    embedding_backend,
    n_hop,
    top_n,
    prune_redundant=True,
    prune_layer_top=True,
    threshold=0.,
    tau=1,
    index_batch_size=10240,
    generate_batch_size=1024,
    embedding_model=None,
    embedding_tokenizer=None
):
    df_QA = get_dataset(dataset_name)
    lst_questions = df_QA["question"].to_list()

    retriever = get_retriever(
        dataset_name=dataset_name,
        embedding_name=embedding_name,
        embedding_backend=embedding_backend,
        model=model.eval(),
        index_batch_size=index_batch_size,
        index_device=DEVICE,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer
    )
    # retriever.search_passages(["Who is "])

    retrieved_result = retriever.multihop_search_passages(
        lst_questions,
        n_hop=n_hop,
        top_n=top_n,
        threshold=threshold,
        tau=tau,
        prune_redundant=prune_redundant,
        prune_layer_top=prune_layer_top,
        index_batch_size=index_batch_size,
        generate_batch_size=generate_batch_size,
    )

    df_match = df_QA.apply(
        match_retrieve,
        retrieved_passages=retrieved_result.passage,
        axis=1,
        result_type="expand"
    )
    df_match = pd.concat([df_QA, df_match], axis=1)
    n_total = df_match["set_evidence_title"].map(len).sum()

    return df_match[1].sum(axis=0) / n_total


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CoRe")

    # multihop retrieval with CoRe
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["hotpotqa_distractor",
                 "2wiki", "musique", "multihop_rag"],
    )
    
    parser.add_argument(
        "--embedding_name", type=str, default="bge-m3",
        help="Name of the embedding to be used"
    )
    parser.add_argument(
        "--embedding_backend", type=str, default="hf",
        help="Backend for embedding retrieval, supports hf or vllm"
    )
    parser.add_argument(
        "--n_hop", type=int, required=True,
        help="Number of hops for multihop retrieval"
    )
    parser.add_argument(
        "--top_n", type=int, required=True,
        help="Number of retrieved chunks for each hop"
    )

    # CoRe model and retrieval settings
    parser.add_argument(
        "--model_name_or_path", nargs='*',
        help="Resume with saved parameters"
    )
    parser.add_argument(
        "--revision", type=str,
        default="main",
        help="Branch name or tag name of the model to be loaded"
    )
    parser.add_argument(
        "--model_cls", nargs='*',
        choices=["infonce", "hoyer"],
        default=["infonce"],
        help="Choose between 'infonce' and 'hoyer', or both for multi-retriever evaluation"
    )
    parser.add_argument(
        "--index_batch_size", type=int, default=10240,
        help="Batch size for Fiass retrieval"
    )
    parser.add_argument(
        "--generate_batch_size", type=int, default=2048,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--prune_redundant", action='store_true',
        help="Toggle stop criterion: redundancy pruning"
    )
    parser.add_argument(
        "--prune_layer_top", action='store_true',
        help="Toggle stop criterion: layer-wise top pruning"
    )
    parser.add_argument(
        "--prune_layer_top_val", type=int, default=None,
        help="Set value of layer-wise top pruning"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Toggle retrieval threshold"
    )
    parser.add_argument(
        "--tau", type=int, default=8,
        help="Toggle retrieval temperature"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print(f"Using {DEVICE}")

    args = parse_args()

    print(f"Evaluating {args.dataset_name} with recall@{args.top_n} under {args.n_hop} hops for {args.model_cls}")
    print(f"Prune redundant: {args.prune_redundant}, prune layer top: {args.prune_layer_top}, threshold: {args.threshold}, tau: {args.tau}")
    lst_embedding_rewriters = []
    for model_name, model_path in zip(args.model_cls, args.model_name_or_path, strict=True):
        if model_name == "infonce":
            from embedding_rewriter.model import InfonceModel
            model_cls = InfonceModel
        elif model_name == "hoyer":
            from embedding_rewriter.hoyer import HoyerModel
            model_cls = HoyerModel
        else:
            raise NotImplementedError(f"Unknown model class: {args.model_name}")

        model = get_embedding_rewriter_model(
            model_path, model_cls,
            device=DEVICE, revision=args.revision,
        )

        lst_embedding_rewriters.append(model)

    df_QA = get_dataset(args.dataset_name)
    retriever = get_retriever(
        dataset_name=args.dataset_name,
        embedding_name=args.embedding_name,
        embedding_backend=args.embedding_backend,
        model=tuple(lst_embedding_rewriters)
    )

    prune_layer_top = args.prune_layer_top_val \
                      if args.prune_layer_top_val is not None \
                      else args.prune_layer_top
    retrieved_result = retriever.multihop_search_passages(
        df_QA["question"].to_list(),
        n_hop=args.n_hop,
        top_n=args.top_n,
        threshold=args.threshold,
        tau=args.tau,
        index_batch_size=args.index_batch_size,
        generate_batch_size=args.generate_batch_size,
        prune_redundant=args.prune_redundant,
        prune_layer_top=prune_layer_top,
        return_tree=True,
        return_query_similarity=True
    )

    retrieved_passages = retrieved_result.passage
    if isinstance(retrieved_result.passage[0][0], dict):
        # For the case of single-hop retrieval
        retrieved_passages = [retrieved_result.passage]

    df_match = df_QA.apply(
        match_retrieve,
        retrieved_passages=retrieved_passages,
        axis=1,
        result_type="expand"
    )
    df_match = pd.concat([df_QA, df_match], axis=1)
    n_total = df_match["set_evidence_title"].map(len).sum()

    print("Iteration recalls:")
    print(df_match[range(args.n_hop)].sum(axis=0).cumsum() / n_total)

    print("Stats by question type:")
    print(
        df_match.groupby(["type", ])[list(range(args.n_hop))].agg(["count", "mean"])
    )

    k = 0.
    for i, psgs in enumerate(retrieved_passages):
        k += sum(map(len, psgs))
        print(f"Avg. K on hop {i+1}:", k / len(psgs) if len(psgs) > 0 else 0)

    with open(f"eval_data/{'_'.join(args.model_cls)}__{args.dataset_name}_result__top_n={args.top_n}&n_hop={args.n_hop}&threshold={args.threshold}&redundant={args.prune_redundant}&layerwise_top={prune_layer_top}.pkl", "wb") as f:
        pickle.dump(retrieved_result, f)
