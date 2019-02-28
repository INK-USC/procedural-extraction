COMMENT=manual1of5
rm -rdf ckpts/$COMMENT
rm -rdf logs/$COMMENT

CUDA_VISIBLE_DEVICES=4 \
python train_bert_context_classifier.py \
--bert_model bert-base-uncased --do_lower_case \
--max_seq_length 512 \
--do_train --do_eval_on_train --do_eval \
--data_dir dataset/manual \
--train_batch_size 6 \
--gradient_accumulation_steps 2 \
--eval_batch_size 6 \
--num_train_epochs 15 \
--learning_rate 1e-5 \
--comment $COMMENT \
--offset_fusion mask