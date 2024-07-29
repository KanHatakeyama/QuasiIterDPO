question_template="以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n"
answer_template="\n\n### 応答:\n"

def gen_prompt(prompt):
    return  f"{question_template}{prompt}{answer_template}"

def ask(prompt,pipe,max_length=512):
    prompt=f"{question_template}{prompt}{answer_template}"
    try:
        ans=pipe(prompt,max_length=max_length)[0]["generated_text"][len(prompt):]
    except:
        return "わかりません"
    return ans


def get_specific_batch(sft_dataset, batch_size, batch_id):
    print("dataset keys:",sft_dataset.features.keys())
    start_index = batch_id * batch_size
    end_index = start_index + batch_size
    if "output" in sft_dataset.features.keys():
        chosen_key="output"
    elif "chosen" in sft_dataset.features.keys():
        chosen_key="chosen"
    else:
        raise ValueError("No output or chosen key found in dataset")

    if "instruction" in sft_dataset.features.keys():
        inst_key="instruction"
    elif "prompt" in sft_dataset.features.keys():
        inst_key="prompt"
    else:
        raise ValueError("No instruction or prompt key found in dataset")

    if start_index >= len(sft_dataset[chosen_key]):
        return []  # If the batch_id is out of range, return an empty list

    batch = {
        "instruction": sft_dataset[inst_key][start_index:end_index],
        #"input": sft_dataset["input"][start_index:end_index],
        "output": sft_dataset[chosen_key][start_index:end_index]
    }
    
    record_list = []
    for j in range(len(batch["instruction"])):
        record = {
            #"prompt": batch["instruction"][j] + batch["input"][j],
            "prompt": batch["instruction"][j] ,
            "chosen": batch["output"][j]
        }
        record_list.append(record)
    
    return record_list

# Example usage:
# specific_batch = get_specific_batch(sft_dataset, batch_size=10, batch_id=2)
