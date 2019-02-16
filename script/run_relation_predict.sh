CUDA_VISIBLE_DEVICES=4 \
python predict_bert_context_classifier.py \
--bert_model bert-base-uncased --do_lower_case \
--max_seq_length 512 \
--data_dir dataset/relation_full \
--batch_size 6 \
--comment postattn \
--offset_fusion postattn