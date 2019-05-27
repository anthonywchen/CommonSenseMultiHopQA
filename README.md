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

We download the raw data for NarrativeQA, MSMarco, and SocialQA into the `raw_data` directory. From the root of the directory, run
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

# Download SocialQA
```

## Build Processed Datasets

We need to build processed datasets. 

For MS Marco (without commonsense information), we run:
```
python src/config.py \
    --mode build_msmarco_dataset \
    --data_dir raw_data/msmarco \
    --processed_dataset_train data/msmarco_train.jsonl \
    --processed_dataset_valid data/msmarco_valid.jsonl \
```

For NarrativeQA (without commonsense information), we run:
```
python src/config.py \
    --mode build_nqa_dataset \
    --data_dir raw_data/narrativeqa \
    --processed_dataset_train data/narrative_qa_train.jsonl \
    --processed_dataset_valid data/narrative_qa_valid.jsonl \
    --processed_dataset_test data/narrative_qa_test.jsonl
```

For SocialQA (without commonsense information), we run:
```
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
```

To train models for NarrativeQA, run:
```
python src/config.py \
    --version baseline_nqa \
    --model_name <model_name> \
    --processed_dataset_train data/narrative_qa_train.jsonl \
    --processed_dataset_valid data/narrative_qa_valid.jsonl \
    --batch_size 24 \
    --max_target_iterations 15 \
```

To train models for SocialQA, run:
```
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
`test_ref0.txt` and `test_ref1.txt`. 

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
```
which generates the output to a new file named `<model_name>\_preds.txt`. 

To score the predictions performance with Rogue-L/BLEU/etc, run
```
python src/eval_generation.py <ref0> <ref1> <output>
```
where `ref0` and `ref1` are the generated reference files for the automatic
metrics. This will also generate a file with the Rogue-L results. 

To generate a file with the BERTScore results, first activate the python3 environment. Then run
```
python src/pycocoevalcap/bert_score/bert_scorer.py \
    --canidate_file <output>
    --reference_file1 <ref0>
    --reference_file2 <ref1>
```

## ToDo 
* Add in SocialQA dataset processing
* Add in SocialQA training
* Clean README, getting read of unncessary info and adding in how to create ELMo embeddings, etc. 
* Also write out BLEU score file
* Script to merge in predictions, metric score, and original data file for easier transferrability
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