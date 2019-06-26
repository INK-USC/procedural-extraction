Code for paper Eliciting Knowledge from Experts: Automatic Transcript Parsing for Cognitive Task Analysis, proceedings of ACL 2019

# Instruction - Install

1. Run install script to download word embeddings, install pre-trained models
   ```
    bash install.sh
   ```
2. Start StandfordCoreNLP Server (please refer to [here](https://stanfordnlp.github.io/CoreNLP/) if your are not familiar with StandfordCoreNLP). This server is required for tokenization in preprocessing and pattern-based extraction.
    ```
    java -mx20g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port 9000
    ```
    You should customize the StandfordCoreNLP Server URL in `pattern_extraction/corenlp.py` by modifying 
    ```
    nlp_server = StanfordCoreNLP('http://YOUR_SERVER:YOUR_PORT')
    ```

# Instruction - Relation Classfication
Step-by-step commands for reproducing our experiment on Relation Classification with setting:
```
Model = Mask_max
Fuzzy-matching Method = Glove 300d
Context-level K = 2
Sampling Portion = 4:2:1
```
Please refer to the ArgumentParser in code for more details. 

## Preprocessed Dataset
Preprocessed Relation Classification Dataset is available in `preprocessed/relation.tar.gz`.

Drop the extracted `relation` folder into `dataset` folder.
Then you can start the [training](#training) of Relation Classfication.

## Preprocessing
**Preprocessed dataset which is available in the [Preprocessed Dataset Section](#preprocessed-dataset).**
...or you can do preprocessing on your own by running following commands...

1. Run preprocessing script. The example script is parsing data from protocol files and doing fuzzy matching with Glove 300d embedding.
   ```
   bash script/run_matching.sh
   ```
2. Create Relation Classification Dataset with the extracted data.
   ```
   python create_dataset.py relation --path dataset/relation/embavg-2 --dir_extracted extracted/embavg
   ```
3. Build Manual Matching Testset, from manual matching annotation `fuzzy_matching/answer.json` on `dataset 1`
   ```
   # Manual Matching
   python extract_samples.py 1 manual --dir_extracted extracted/manual
   # Create Relation Classification Dataset for context level K=2
   python create_dataset.py relation --path dataset/relation/manual-2 --k_neighbour 1 --dir_extracted extracted/manual --dataset 1
   # Copy to Fuzzy Matching dataset as its manual-matching testset
   cp dataset/relation/manual-2/nonsplit.pkl dataset/relation/embavg-2/manual.pkl
   ``` 
## Training
1. Train Matching Model for 5 times
   ```
   bash run_relation.sh
   ```
2. Get Averaged Result
   ```
   python script/reduce.py maskmax-k2
   cat logs/maskmax-k2-avg/metrics.json
   ```

# Instruction - Sequence Labeling
You can also create sequence labeling dataset with the fuzzy-matching result:
```
mkdir -p dataset/seqlab/embavg
python create_dataset.py seqlabel --dir_extracted extracted/embavg --path dataset/seqlab/embavg
```
SeqLab dataset with IOBES format is created in folder `dataset/seqlab/embavg`

# Limitation
Manual-matching testset is created on `dataset 1`, so a small portion of samples in the manual-matching testset would be overlapping with samples in fuzzy-matching trainset (though they are extracted by different methods and are resampled). This is a flaw of the manual testset setting.
