 ds='1 2 3 4 5'
 for d in $ds
 do
    COMMENT=mask
    rm -rdf ckpts/$COMMENT-$d
    rm -rdf logs/$COMMENT-$d

    CUDA_VISIBLE_DEVICES=1 \
    python train_bert_context_classifier.py \
    --bert_model bert-base-uncased --do_lower_case \
    --max_seq_length 512 \
    --do_train --do_eval_on_train --do_eval --do_manual \
    --data_dir dataset/new_embavg \
    --train_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --eval_batch_size 4 \
    --num_train_epochs 15 \
    --learning_rate 1e-5 \
    --comment $COMMENT-$d \
    --offset_fusion mask
    --seed $d
done