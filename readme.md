# Procedural Extraction
Code for paper *Eliciting Knowledge from Experts: Automatic Transcript Parsing for Cognitive Task Analysis*, in proceedings of ACL 2019 

This code provides a framwork for extracting procedural information from documents. Please refer to our ACL paper (uploading to arXiv) for further descriptions.

# Quick Links
* [Install](#install)
* [Run Relation Classification](#run-relation-classfication)
* [Run Sequence Labeling](#run-sequence-labeling)
* [Reported Numbers](#reported-numbers)
* [Data](#data)
* [Cite](#cite)

# Install

1. Run install script to download word embeddings, install pre-trained models
   ```bash
    bash install.sh
   ```
2. Start StandfordCoreNLP Server (please refer to [here](https://stanfordnlp.github.io/CoreNLP/) if your are not familiar with StandfordCoreNLP). This server is required for tokenization in preprocessing and pattern-based extraction.
    ```bash
    java -mx20g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port 9000
    ```
    You should customize the StandfordCoreNLP Server URL in `pattern_extraction/corenlp.py` by modifying 
    ```python
    nlp_server = StanfordCoreNLP('http://YOUR_SERVER:YOUR_PORT')
    ```

# Run Relation Classfication
Step-by-step commands for running experiment on Relation Classification with setting:
* Model = Mask_max
* Fuzzy-matching Method = Glove 300d
* Context-level K = 2
* Sampling Portion = 4:2:1

Please refer to the ArgumentParser in code for more details. 

## Preprocessed Dataset
Preprocessed Relation Classification Dataset is available in `preprocessed/relation.tar.gz`.

Drop the extracted `relation` folder into `dataset` folder.
Then you can run the [Model](#model) of Relation Classfication.

## Preprocessing
**Preprocessed dataset is available in the [Preprocessed Dataset Section](#preprocessed-dataset).** 
...or you can do preprocessing on your own by running following commands...

1. Run preprocessing script. The example script is parsing data from protocol files and doing fuzzy matching with Glove 300d embedding.
   ```bash
   bash script/run_matching.sh
   ```
2. Create Relation Classification Dataset with the extracted data.
   ```bash
   python create_dataset.py relation --path dataset/relation/embavg-2 --dir_extracted extracted/embavg
   ```
3. Build Manual Matching Testset, from manual matching annotation `fuzzy_matching/answer.json` on `document 01`
   ```bash
   # Manual Matching
   python extract_samples.py 1 manual --dir_extracted extracted/manual
   # Create Relation Classification Dataset for context level K=2
   python create_dataset.py relation --path dataset/relation/manual-2 --k_neighbour 1 --dir_extracted extracted/manual --dataset 1
   # Copy to Fuzzy Matching dataset as its manual-matching testset
   cp dataset/relation/manual-2/nonsplit.pkl dataset/relation/embavg-2/manual.pkl
   ``` 
## Model
1. Train Matching Model for 5 times
   ```bash
   bash run_relation.sh
   ```
2. Get Averaged Result
   ```bash
   python script/reduce.py maskmax-k2
   cat logs/maskmax-k2-avg/metrics.json
   ```

# Run Sequence Labeling
## Preprocessing
You can also create sequence labeling dataset with the fuzzy-matching result:
```bash
mkdir -p dataset/seqlab/embavg
python create_dataset.py seqlabel --dir_extracted extracted/embavg --path dataset/seqlab/embavg
```
SeqLab dataset with IOBES format is created in folder `dataset/seqlab/embavg`

## Model
Please refer to [Standford-NER](https://nlp.stanford.edu/software/CRF-NER.html) and [LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF)

# Reported Numbers
## Relation Classfication
| Setting                             | Generated Acc. | Generated F1 | Manual Acc. | Manual F1  |
|-------------------------------------|----------------|--------------|-------------|------------|
| BERT                                | 81.6 ± 1.0     | 70.1 ± 1.7   | 77.2 ± 2.7  | 62.2 ± 6.1 |
| Context position as Attention       | 82.5 ± 1.5     | **72.2 ± 2.6**   | 81.2 ± 4.7  | 72.7 ± 7.5 |
| Context position as Input Embedding | **82.8 ± 1.4**     | 72.7 ± 1.9   | 78.8 ± 8.5  | 67.4 ± 8.1 |
| Hidden States Masking_avg pooling   | 80.5 ± 2.7     | 69.0 ± 5.7   | 80.4 ± 7.1  | 73.4 ± 7.9 |
| Hidden States Masking_max pooling   | 82.3 ± 1.4     | 72.6 ± 3.0   | **87.6 ± 1.5**  | **81.4 ± 2.4** |

* Fuzzy-matching Method = Glove 300d
* Context-level K = 2
* Sampling Portion = 4:2:1

# Data
6 documents in folder `data` for creating dataset
1. ***.src.txt:** Copy from original source (transcript) files (seperated by \n)
2. ***.src.ref.txt:** Line-by-line extracted by utils/copyLineByLine.vbs from original source files, with line-break shown in Microsoft Word
3. ***.tgt.txt:** Original target (protocol) files (seperated by \n)

`document 05` is not used in our experiments since the line numbers in protocol is not accurate.

Manual-matching testset is created on `document 01`, so a small portion of samples in the manual-matching testset would be overlapping with samples in fuzzy-matching trainset (though they are extracted by different methods and are resampled). This is a flaw of the manual testset setting.

# Cite
```
@inproceedings{du2019acl,
  title={Eliciting Knowledge from Experts: Automatic Transcript Parsing for Cognitive Task Analysis},
  author={Junyi, Du and He, Jiang and Jiaming, Shen and Xiang, Ren},
  booktitle={Proceedings of ACL 2019},
  year={2019}
}
```
