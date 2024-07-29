# %%
import argparse
from datasets import load_dataset,load_from_disk
from vllm import SamplingParams, LLM
from src.utils import gen_prompt, get_specific_batch
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Script to generate predictions using a specified model and dataset")
    parser.add_argument("--generation", type=int, default=0, help="Generation number")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--model_id", type=str, default="team-hatakeyama-phase2/8B-nishijima-tanuki8b_dpo_full_001-checkpoint-137", help="Model ID")
    parser.add_argument("--sft_dataset_name", type=str, default="kanhatakeyama/0717-calm3-22b-random-genre-inst-sft-tsub-part", help="SFT dataset name")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="split")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum model length")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    args = parser.parse_args()
    return args

def ask_vllm(llm, prompts, max_tokens, repetition_penalty):
    outputs = llm.generate(
        prompts,
        sampling_params=SamplingParams(
            temperature=0.01,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )
    )   
    return [(output.outputs[0].text).strip() for output in outputs]

def main():
    args = parse_args()
    
    out_jsonl_path = f"{args.out_dir}/{args.generation}.jsonl"
    
    llm = LLM(model=args.model_id, trust_remote_code=True, max_model_len=args.max_model_len)
    
    try:
        sft_dataset = load_dataset(args.sft_dataset_name, split=args.split)
    except:
        sft_dataset = load_from_disk(args.sft_dataset_name )
    
    sft_dataset=sft_dataset.shuffle()
    record_list = get_specific_batch(sft_dataset, batch_size=args.batch_size, batch_id=args.generation)
    
    prompt_list = [gen_prompt(r["prompt"]) for r in record_list]
    predictions = ask_vllm(llm, prompt_list, args.max_tokens, args.repetition_penalty)
    
    for record, pred in zip(record_list, predictions):
        if pred=="":
            pred="分かりません。"
        record["rejected"] = pred
        
    with open(out_jsonl_path, "w") as f:
        for record in record_list:
            if record["chosen"] != record["rejected"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

# %%
