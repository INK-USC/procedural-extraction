 ds='1 2 3 4 6'
 for d in $ds
 do
    CUDA_VISIBLE_DEVICES=6 \
    python extract_samples.py $d ex-bert \
    --output \
    --dir_extracted extracted/exbert \
    --bert_model bert-base-uncased --do_lower_case --batch_size 8 --sum_four
done