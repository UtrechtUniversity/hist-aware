python transformers/examples/pytorch/language-modeling/run_mlm.py \
    --tokenizer_name /home/leonardovida/data/volume_1/data-histaware/tokenizer/1970/dutch.bert.vocab_mod.128.cased.json \
    --cache_dir /home/leonardovida/data/volume_1/huggingface_cache \
    --train_file /home/leonardovida/data/volume_1/data-histaware/merged_articles/1970s/merged_articles/test/2.txt \
    --output_dir /home/leonardovida/data/volume_1/data-histaware/model \
    --logging_dir /home/leonardovida/data/volume_1/data-histaware/logging \
    --model_type bert \
    --max_seq_length 128 \
    --do_train \
    --preprocessing_num_workers 8 \
    --mlm_probability 0.15 \
    --pad_to_max_length True \
    --overwrite_cache True \
    --overwrite_output_dir True
    
    
    --evaluation_strategy steps \
    --per_device_train_batch_size 4 \
    --logging_strategy steps \
    --logging_steps 700 \
    --save_total_limit 3 \
    --seed 2909 \
    --prediction_loss_only True \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 0 \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --save_steps -1