 CUDA_VISIBLE_DEVICES=0 \
 python extract_samples.py 1 ex-bert \
 --dir_extracted extracted/bert \
 --bert_model bert-base-uncased --do_lower_case --batch_size 32