export GLUE_DIR=/data/junyi/glue_data

python train_bert_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir checkpoints \
  --max_seq_length 64 \
  --eval_batch_size 32