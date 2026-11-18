import os
os.chdir(os.path.join(os.path.abspath("."), ".."))
import pandas as pd
from evaluation import get_dataset


model_cls = "hoyer"
for dataset_name in ["hotpotqa_distractor", "2wiki", "musique", "multihop_rag"]:
    df_QA = get_dataset(dataset_name)
    for top_n, n_hop in [(5, 3), (10, 2)]:
        retrieved_result = pd.read_pickle(f"eval_data/{model_cls}__{dataset_name}_result__top_n={top_n}&n_hop={n_hop}&threshold=0.0&redundant=True&layerwise_top=True.pkl")

        retrieved_passages = retrieved_result.passage
        if isinstance(retrieved_result.passage[0][0], dict):
            # For the case of single-hop retrieval
            retrieved_passages = [retrieved_result.passage]

        assert len(retrieved_passages) == n_hop, \
            f"Expected {n_hop} hops, but got {len(retrieved_passages)}"
        assert len(retrieved_passages[0]) == len(df_QA), \
            f"Expected {len(df_QA)} questions, but got {len(retrieved_passages[0])}"

        for i in range(len(retrieved_passages)):
            df_QA[f"recall@{top_n}_hop{i+1}"] = retrieved_passages[i]
            if i > 0:
                df_QA[f"recall@{top_n}_hop{i+1}"] = df_QA.apply(
                    lambda df: df[f"recall@{top_n}_hop{i}"] + df[f"recall@{top_n}_hop{i+1}"],
                    axis=1
                )

    df_QA.to_json(f"eval_data/{dataset_name}_dev_MHQA.jsonl",
                orient="records", lines=True)


def prepare_iter_retgen(df, n_hop):
    inter = df["inter"]
    lst_res = [[]] * n_hop
    for i in range(len(inter)):
        lst_res[i] = inter[i]["docs"]
        if i > 0:
            lst_res[i] = lst_res[i-1] + lst_res[i]

    return lst_res


top_n = 5
n_hop = 3
# for model in ["qwen2.5-70B", "llama3.1-8B"]:
for model in ["qwen3-0.6B"]:
    for dataset_name in ["hotpotqa_distractor", "2wiki", "musique", "multihop_rag"]:
        df_QA = pd.read_json(f"eval_data/iter_retgen_{model}/{dataset_name}.jsonl",
                            orient="records", lines=True)

        df_res = df_QA.apply(prepare_iter_retgen, n_hop=3, result_type='expand', axis=1)
        df_QA[[f"recall@{top_n}_hop{i+1}" for i in range(n_hop)]] = df_res

        df_data = pd.read_json(f"eval_data/{dataset_name}_dev_processed.jsonl",
                                orient="records", lines=True)
        df_QA = df_data[["question", "answer", "type", "supporting_facts"]].merge(
                df_QA,
                on="question", how="right")
        df_QA.to_json(f"eval_data/iter_retgen_{model}/{dataset_name}_dev_MHQA.jsonl",
                orient="records", lines=True)