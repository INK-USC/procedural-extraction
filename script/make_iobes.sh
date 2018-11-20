DEVN=$(printf "%02d" $2)
TESTN=$(printf "%02d" $3)
DATA_SET=(01 02 03 04 05 06)
SUFFIX=$1.iobes
VANILLA_NER=false
VANILLA_NER_PATH=/data/junyi/github/Vanilla_NER/data/iobes
LSTM_CRF_PATH=/data/junyi/github/LM-LSTM-CRF/data
CORE_NLP_PATH=/data/junyi/corenlp

TRAIN_SET=()

for i in ${DATA_SET[*]}; do
    python AP_extraction/create_iobes.py $i $1
    if [[ $i == $DEVN ]]; then
        DEV_SET=$DEVN.$SUFFIX
    elif  [[ $i == $TESTN ]]; then
        TEST_SET=$TESTN.$SUFFIX
    else
        TRAIN_SET+=($i.$SUFFIX)
    fi
done

echo TRAIN_SET: ${TRAIN_SET[*]}
echo DEV_SET: ${DEV_SET[*]}
echo TEST_SET: ${TEST_SET[*]}

cd data

cat ${TRAIN_SET[*]} > ../train$DEVN$TESTN.iobes
cat ${DEV_SET[*]} > ../dev$DEVN$TESTN.iobes
cat ${TEST_SET[*]} > ../test$DEVN$TESTN.iobes

cd ../
echo 'distributing iobes files'
ls | grep .iobes
cp *.iobes $VANILLA_NER_PATH
cp *.iobes $LSTM_CRF_PATH
mv *.iobes $CORE_NLP_PATH

if ($VANILLA_NER); then
    cd /data/junyi/github/Vanilla_NER/
    echo 'creating Vanilla NER mapping'
    python pre_seq/gene_map.py --train_corpus ./data/iobes/train.iobes --threshold 0
    echo 'creating Vanilla NER dataset'
    python pre_seq/encode_data.py --train_file ./data/iobes/train$DEV_SET$TEST_SET.iobes --test_file ./data/iobes/test$DEV_SET$TEST_SET.iobes --dev_file ./data/iobes/dev$DEV_SET$TEST_SET.iobes
    echo 'removing Vanilla NER last checkpoint'
    rm checkpoint/ner -rd
fi