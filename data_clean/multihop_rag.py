import os
import pandas as pd
os.chdir(os.path.join(os.path.abspath("."), ".."))


df_QA = pd.read_json("hf://datasets/yixuantt/MultiHopRAG/MultiHopRAG.json")
df_QA = df_QA.rename(columns={"question_type": "type",
                              "query": "question",
                              "evidence_list": "evidences"})
df_QA["supporting_facts"] = df_QA["evidences"].apply(
    lambda lst: [(evd["title"], i) for i, evd in enumerate(lst)]
)
df_QA.to_json("eval_data/multihop_rag_dev_processed.jsonl", lines=True, orient="records")