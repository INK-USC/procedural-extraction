PAT=../procedural-extraction/dataset/seqlab
GPU=6

python train_wc.py \
--word_layers 2 \
--train_file $PAT/train.iobes \
--dev_file $PAT/dev.iobes \
--test_file $PAT/test.iobes \
--caseless --high_way --least_iters 100 \
--gpu $GPU --fine_tune \
--batch_size 24 \
--lr 0.045 --drop_out 0.3 \
--patience 30 \
--mini_count 1 \
--word_hidden 256 \
--char_hidden 16