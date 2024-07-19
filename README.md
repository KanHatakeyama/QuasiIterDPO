
~~~
batch_size=2048
dataset_name=kanhatakeyama/0717-calm3-22b-random-genre-inst-sft-tsub-part
initial_model=team-hatakeyama-phase2/8B-nishijima-tanuki8b_dpo_full_001-checkpoint-137

#############
#generation 0
#reject dataの生成
python gen_rejected.py --generation 0 --batch_size $batch_size --model_id $initial_model --sft_dataset_name $dataset_name
#dpo
python run_dpo.py --generation 0 --model_id $initial_model

################3
# generation 1
python gen_rejected.py --generation 1 --batch_size $batch_size --model_id out/0 --sft_dataset_name $dataset_name
python run_dpo.py --generation 1 --model_id out/0

#generation 2
python gen_rejected.py --generation 2 --batch_size $batch_size --model_id out/1 --sft_dataset_name $dataset_name
python run_dpo.py --generation 1 --model_id out/1
~~~