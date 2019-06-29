# Commonsense for Generative Multi-Hop Question Answering Tasks (EMNLP 2018)

This repository contains the code and setup instructions for our EMNLP 2018 paper
"Commonsense for Generative Multi-Hop Question Answering Tasks". See full paper
[here](https://arxiv.org/abs/1809.06309).

## Environment Setup

We trained our models with python 2 and TensorFlow 1.3, a full list of python
packages is listed in `requirements.txt`

To use BERTScore, you also need use python3 and install the python packages in 
`python3_requirements.txt`

## Downloading Data
First run `./setup.sh` to set up the directory structure. 

We download the raw data for NarrativeQA, MSMarco, and SemEval-2018 Task 11 into the `raw_data` directory. From the root of the directory, run
```
cd raw_data

# Download NarrativeQA
git clone https://github.com/deepmind/narrativeqa.git
cd ../

# Download MSMarco
mkdir msmarco
cd msmarco
wget https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz
gunzip -r train_v2.1.json.gz
gunzip -r dev_v2.1.json.gz
cd ../

# Download SemEval-2018 Task 11
mkdir semeval
cd semeval
wget https://raw.githubusercontent.com/DungLe13/commonsense/master/data/train-data.xml
wget https://raw.githubusercontent.com/DungLe13/commonsense/master/data/dev-data.xml
cd ..

```

## Build Processed Datasets

We need to build processed datasets. 

For MS Marco (without commonsense information), we run:
```
python src/config.py \
    --mode build_msmarco_dataset \
    --data_dir raw_data/msmarco \
    --processed_dataset_train data/msmarco/msmarco_train.jsonl \
    --processed_dataset_valid data/msmarco/msmarco_valid.jsonl \
```

For NarrativeQA (without commonsense information), we run:
```
python src/config.py \
    --mode build_nqa_dataset \
    --data_dir raw_data/narrativeqa \
    --processed_dataset_train data/nqa/narrative_qa_train.jsonl \
    --processed_dataset_valid data/nqa/narrative_qa_valid.jsonl \
    --processed_dataset_test data/nqa/narrative_qa_test.jsonl
```

For SemEval-2018 Task 11 (without commonsense information), we run:
```
python src/config.py \
    --mode build_semeval_dataset \
    --data_dir raw_data/semeval \
    --processed_dataset_train data/semeval/semeval_train.jsonl \
    --processed_dataset_valid data/semeval/semeval_valid.jsonl
```

## Generating ELMo vocab and embeddings 
The ELMo vocab file containing the vocabulary for each dataset and the associated ELMo word embeddings should be computed for each dataset before training. We can donwload the ELMo files for NarrativeQA but will have to generate the ELMo files for the other datasets.

First download the pre-computed ELMo representation for NarrativeQA.  [here](https://drive.google.com/file/d/1pwzyEa0ogrXAMDmkFWOwH_eCSk8bP7ud/view), and extract into the folder `lm_data`. This contains the ELMo data for NarrativeQA. 

To extract the ELMo files, run the following in the `lm_data` file. 
```
## For MSMarco
# Write vocabulary file
python src/write_vocabulary.py \
    --processed_dataset_train  data/msmarco/msmarco_train.jsonl \
    --processed_dataset_valid   data/msmarco/msmarco_valid.jsonl \
    --output_vocabulary  lm_data/msmarco/msmarco_vocab.txt 

# Generate ELMo embeddings
python src/write_elmo_embeddings.py \
    --vocab_file lm_data/msmarco/msmarco_vocab.txt \
    --hdf5_output_file lm_data/semeval/elmo_token_embeddings-msmarco.hdf5
    
## For SemEval-2018 Task 11
# Write vocabulary file
python src/write_vocabulary.py \
    --processed_dataset_train  data/semeval/semeval_train.jsonl \
    --processed_dataset_valid   data/semeval/semeval_valid.jsonl \
    --output_vocabulary  lm_data/semeval/semeval_vocab.txt 

# Generate ELMo embeddings
python src/write_elmo_embeddings.py \
    --vocab_file lm_data/semeval/semeval_vocab.txt \
    --hdf5_output_file lm_data/semeval/elmo_token_embeddings-msmarco.hdf5

```

## Training & Evaluation

### Training

To train models for MS Marco, run:
```
python src/config.py \
    --version baseline_nqa \
    --model_name <model_name> \
    --processed_dataset_train data/msmarco_train.jsonl \
    --processed_dataset_valid data/msmarco_valid.jsonl \
    --elmo_token_embedding_file lm_data/msmarco/elmo_token_embeddings.hdf5 \
    --elmo_vocab_file lm_data/msmarco/msmarco_vocab.txt \
    --num_epochs 8 \
    --batch_size 36 \
    --dropout_rate 0.2
```

To train models for NarrativeQA, run:
```
python src/config.py \
    --version baseline_nqa \
    --model_name <model_name> \
    --processed_dataset_train data/nqa/narrative_qa_train.jsonl \
    --processed_dataset_valid data/nqa/narrative_qa_valid.jsonl \
    --batch_size 24 \
    --max_target_iterations 15 \
    --dropout_rate 0.2 
```

To train models for SemEval-2018 Task 11, run:
```
python src/config.py \
    --version baseline_nqa \
    --model_name <model_name> \
    --processed_dataset_train data/semeval/semeval_train.jsonl \
    --processed_dataset_valid data/semeval/semeval_valid.jsonl \
    --batch_size 32 \
    --max_target_iterations 15 \
    --num_epochs 12 \
    --dropout_rate 0.2 
```

### Evaluation
We will demonstrate how to evaluate using NarrativeQA as an example, but the commands are analgous for the other dataset.

To evaluate NarrativeQA, we need to first generate official answers on the test
set. To do so, run:
```
python src/config.py \
    --mode generate_answers \
    --processed_dataset_valid data/narrative_qa_valid.jsonl \
    --processed_dataset_test data/narrative_qa_test.jsonl 
```

This will create the reference files `val_ref0.txt`, `val_ref1.txt`,
`test_ref0.txt` and `test_ref1.txt`. Move these files into the `data/nqa` directory. 

To generate predictions on the dev/test set using the trained model, run
```
python src/config.py \
    --mode test \
    --version baseline_nqa \
    --model_name <model_name> \
    --use_ckpt <ckpt_name> \
    --use_dev False \ # False to evaluate test set, True to evaluate dev set.
    --processed_dataset_train data/narrative_qa_train.jsonl \
    --processed_dataset_valid data/narrative_qa_valid.jsonl \
    --processed_dataset_test data/narrative_qa_test.jsonl \
    --batch_size 24 \
    --max_target_iterations 15 \
    --dropout_rate 0.2 
```
which generates the output (a new file named <model_name>\_preds.txt). 

To score the predictions performance with Rogue-L/BLEU/etc, run
```
python src/eval_generation.py <ref0> <ref1> <output>
```
where `ref0` and `ref1` are the generated reference files for the automatic
metrics. This will also generate a files with sentence level scores for BLEU, 
Rouge, Meteor, and Cider scores. 


To generate a file with the BERTScore results, first activate the python3 environment. Then run
```
python src/pycocoevalcap/bert_score/bert_scorer.py \
    --candidates_file <output> \
    --references_file1 <ref0> \
    --references_file2 <ref1>
```

To generate a file with the paraphrase score with a model trained using the 
`Paraphrase-Identification` repo, first activate the python3 environment. Then run
```
python src/pycocoevalcap/paraphrase_score/paraphrase_scorer.py \
    --paraphrase_identification_repo <path to the paraphrase identification repository> \
    --paraphrase_archive <path to the trained model archive in the parapharse identification repo> \
    --candidates_file <output> \
    --references_file1 <ref0> \
    --references_file2 <ref1> \
    --cuda_device [0, -1] # Set 0 to use first GPU, -1 for CPU
```

Merge the data file, predictions file, and the different scoring files (of the test set)
into one file for easy portability. Assume that `out/nqa_baseline` is the directory
of the trained model, run
```
python src/merge_data_predictions_scores.py \
    --data_file data/nqa/narrative_qa_test.jsonl \
    --prediction_file out/nqa_baseline/test_preds.txt \
    --bleu1_file out/nqa_baseline/test_preds.txt-bleu1.txt \
    --bleu4_file out/nqa_baseline/test_preds.txt-bleu4.txt \
    --rouge_file out/nqa_baseline/test_preds.txt-rougeL.txt \
    --meteor_file out/nqa_baseline/test_preds.txt-meteor.txt \
    --cider_file out/nqa_baseline/test_preds.txt-cider.txt \
    --bert_score_file out/nqa_baseline/test_preds.txt-bert_score.txt \
    --paraphrase_score_file out/nqa_baseline/test_preds.txt-paraphrase_score.txt
```

## ToDo 
* Add in SocialQA dataset processing
* Add in SocialQA training
* Clean README, getting read of unncessary info and adding in how to create ELMo embeddings, etc. 
* Create paraphrase detection model to use as additional metric. 

## Bibtex
```
@inproceedings{bauerwang2019commonsense,
  title={Commonsense for Generative Multi-Hop Question Answering Tasks},
  author={Lisa Bauer*, Yicheng Wang* and Mohit Bansal},
  booktitle={Proceedings of the Empirical Methods in Natural Language Processing},
  year={2018}
}
```
