rm checkpoints/*
CUDA_VISIBLE_DEVICES=1 \
python train_bert_context_classifier.py \
--bert_model bert-base-uncased --do_lower_case \
--output_dir checkpoints \
--max_seq_length 256 --do_train --do_eval \
--data_dir dataset/relation \
--task_name rel \
--train_batch_size 12