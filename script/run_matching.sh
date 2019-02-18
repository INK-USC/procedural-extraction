 ds='1 2 3 4 6'
 for d in $ds
 do
    #CUDA_VISIBLE_DEVICES=2 \
    python extract_samples.py $d embavg \
    --output \
    --dir_extracted extracted/embavg
    #--bert_model bert-base-uncased --do_lower_case --batch_size 16 --mle
done