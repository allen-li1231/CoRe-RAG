import os
import pandas as pd
os.chdir(os.path.join(os.path.abspath("."), ".."))


def gen_evidences(df):
    lst_evidences = []
    ctxs = df["ctxs"]
    for i, evi in enumerate(df["question_decomposition"]):
        support_ctx = ctxs[evi["paragraph_support_idx"]]
        lst_evidences.append((support_ctx["title"], i))

    return lst_evidences

splits = {'train': 'musique_ans_v1.0_train.jsonl', 'validation': 'musique_ans_v1.0_dev.jsonl'}
df_QA = pd.read_json("hf://datasets/bdsaglam/musique/" + splits["validation"], lines=True)

df_QA["type"] = df_QA["id"].str.extract(r"(\dhop\d*)__.*")
df_QA["ctxs"] = df_QA["paragraphs"].apply(lambda ctxs: [{"title": ctx["title"], "text": ctx["paragraph_text"]} for ctx in ctxs])
df_QA["supporting_facts"] = df_QA.apply(gen_evidences, axis=1)
df_QA.drop(columns=["paragraphs", "answerable"], inplace=True)
df_QA.to_json("eval_data/musique_dev_processed.jsonl", lines=True, orient="records")