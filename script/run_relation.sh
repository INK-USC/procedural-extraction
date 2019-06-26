 seed='1 2 3 4 5'
 COMMENT=maskmax-k2
 for d in $seed
 do
    rm -rdf ckpts/$COMMENT-$d
    rm -rdf logs/$COMMENT-$d

    CUDA_VISIBLE_DEVICES=3 \
    python train_bert_context_classifier.py \
    --bert_model bert-base-uncased --do_lower_case \
    --max_seq_length 512 \
    --do_train --do_eval_on_train --do_eval --do_manual \
    --data_dir /home/ron_data/junyi/procedural-extraction/dataset/relation/embavg-2 \
    --train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --eval_batch_size 6 \
    --num_train_epochs 15 \
    --learning_rate 1e-5 \
    --comment $COMMENT-$d \
    --offset_fusion mask \
    --seed $d
done