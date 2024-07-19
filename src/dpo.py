from transformers import TrainingArguments
from trl import DPOTrainer


def run_dpo(model,tokenizer,dpo_dataset,
        run_name="test"):
    training_args = TrainingArguments(
        output_dir= f"./{run_name}",
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        learning_rate=5.0e-7,
        warmup_ratio=0.1,
        num_train_epochs=1,
        logging_steps=1,
        #eval_strategy="steps",
        #eval_steps=200,
        #save_strategy="epoch",
        remove_unused_columns=False,
        bf16=True,
        dataloader_num_workers=24,
        weight_decay = 0.0,
        lr_scheduler_type="cosine",
        gradient_checkpointing=False,
        run_name=run_name,
        #report_to="wandb",
        optim="adamw_torch",
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=0.1,
        loss_type="sigmoid",
        max_prompt_length=925,
        max_length=1150,
        train_dataset=dpo_dataset,
        #eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()