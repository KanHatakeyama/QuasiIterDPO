# %%
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dpo import run_dpo
import json
import argparse
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Script to run DPO with a specified model and dataset")
    parser.add_argument("--generation", type=int, help="Generation number")
    parser.add_argument("--model_id", type=str, default="team-hatakeyama-phase2/8B-nishijima-tanuki8b_dpo_full_001-checkpoint-137", help="Model ID")
    parser.add_argument("--in_jsonl_path", type=str,  help="Input JSONL path")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--wandb_mode", type=str, default="offline", help="WandB mode (offline/online)")
    parser.add_argument("--run_name", type=str, default="test", help="WandB run name")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.in_jsonl_path is None:
        args.in_jsonl_path = f"{args.out_dir}/{args.generation}.jsonl"
    
    wandb.init(project="wandb_test", anonymous="allow", mode=args.wandb_mode)
    
    with open(args.in_jsonl_path, "r") as f:
        new_ds = [json.loads(line) for line in f]
    
    dpo_dataset = Dataset.from_list(new_ds)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    run_dpo(model, tokenizer, dpo_dataset, run_name=args.run_name)
    
    model.save_pretrained(f"{args.out_dir}/{args.generation}")
    tokenizer.save_pretrained(f"{args.out_dir}/{args.generation}")

if __name__ == "__main__":
    main()

# %%
