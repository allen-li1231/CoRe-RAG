import functools
import pandas as pd


def model_file_name_to_params(name):
    lst_params = name.rstrip(".pt").split('__')[1].split('&')
    d_params = dict([param.split('=') for param in lst_params])
    if "dropout" in d_params:
        d_params["dropout"] = float(d_params["dropout"])
    if "weight_decay" in d_params:
        d_params["weight_decay"] = float(d_params["weight_decay"])
    if "lr" in d_params:
        d_params["lr"] = float(d_params["lr"])
    if "temperature" in d_params:
        d_params["temperature"] = float(d_params["temperature"])
    if "batch_size" in d_params:
        d_params["batch_size"] = int(d_params["batch_size"])
    if "n_neg" in d_params:
        d_params["n_neg"] = int(d_params["n_neg"])
    if "x_size" in d_params:
        d_params["x_size"] = int(d_params["x_size"])
    if "g_size" in d_params:
        d_params["g_size"] = int(d_params["g_size"])
    if "mlp_size" in d_params:
        d_params["mlp_size"] = int(d_params["mlp_size"])
    if "n_mlp" in d_params:
        d_params["n_mlp"] = int(d_params["n_mlp"])
    if "n_head" in d_params:
        d_params["n_head"] = int(d_params["n_head"])

    return d_params


@functools.lru_cache()
def get_embedding_rewriter_model(state_dict: str, model_cls, device="cpu", **hf_model_kwargs):
    d_params = model_file_name_to_params(state_dict)
    model = model_cls(
        x_size=1024,
        g_size=int(d_params["g_size"]),
        mlp_size=int(d_params["mlp_size"]),
        n_mlp=int(d_params["n_mlp"]),
        n_head=int(d_params["n_head"])
    )

    import torch
    pt_state_dict = torch.load(state_dict, weights_only=True, map_location=device)
    model.load_state_dict(pt_state_dict)
    model.to(device).compile()
    return model


@functools.lru_cache()
def get_dataset(dataset_name):
    df_QA = pd.read_json(f"eval_data/{dataset_name}_dev_processed.jsonl", lines=True)
    # df_QA = (df_QA[~df_QA["type"].isin(["comparison", # 2wiki
    #                                     # multihop_rag
    #                                     "comparison_query", "null_query", "temporal_query"
    #                                     ])]
    #          .reset_index())
    df_QA["set_evidence_title"] = df_QA["supporting_facts"].apply(
        lambda lst: set([evd[0] for evd in lst])
    )
    return df_QA
