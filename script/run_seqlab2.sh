PAT=../procedural-extraction/dataset/seqlab
GPU=5

python train_w.py \
--train_file $PAT/train.iobes \
--dev_file $PAT/dev.iobes \
--test_file $PAT/test.iobes \
--caseless --least_iters 100 \
--gpu $GPU \
--batch_size 24 \
--lr 0.045 --drop_out 0.55 \
--layers 2 \
--hidden 256 \
--patience 30 \
--mini_count 2