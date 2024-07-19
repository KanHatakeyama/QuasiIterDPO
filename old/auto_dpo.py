
import os
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from tqdm import tqdm
from src.utils import ask,get_specific_batch
from src.dpo import run_dpo
import json
import os

# %%
import wandb
mode = 'offline'
wandb.init(project="wandb_test",
           anonymous="allow",
		   mode=mode)

# %%
out_dataset_dir="out"

start_batch=0
os.system(f"mkdir -p {out_dataset_dir}")

# %%

# Load model and tokenizer
model_id = "hatakeyama-llm-team/Tanuki-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
pipe=pipeline("text-generation",model=model,tokenizer=tokenizer)

# %%
sft_dataset=load_dataset("kanhatakeyama/0717-calm3-22b-random-genre-inst-sft-tsub-part",split="train")

# %%
batch_size=256
batch_size=16
max_length=1024

# %%

#バッチからデータを取得
for batch_id in range(100):
    ds=get_specific_batch(sft_dataset, batch_size=batch_size, batch_id=batch_id+start_batch)

    question_list=[record["prompt"] for record in ds]

    #生徒モデルによる出力の生成
    rejected_ans_list=[ask(prompt,pipe,max_length=max_length) for prompt in tqdm(question_list)]


    #rejectedを反映したデータセットの生成
    new_ds=[]
    for record,rejected in zip(ds,rejected_ans_list):
        record["rejected"]=rejected
        record["batch_id"]=batch_id
        if record["rejected"]!=record["chosen"]:
            new_ds.append(record)

    #書き出し
    with open(f"{out_dataset_dir}/batch.json","a") as f:
        for record in new_ds:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")

    dpo_dataset=Dataset.from_list(new_ds)

    #dpo
    run_dpo(model,tokenizer,dpo_dataset,
            run_name="test")

# %%


# %%



