# MetaGE

## Environment Setup

- Create python environment
    - `conda create -n MetaGE python=3.10.13`
    - `conda activate MetaGE`
    - `pip install pip==24.0`
    - `pip install omegaconf==2.0.5`
    - `pip install fairseq==0.12.2`
    - `pip install -r requirements.txt`

## Using MetaGE for metagenomic tasks

### virus_identification

    Step 1: Preprocess your data

    - `python src/pre-process.py --data_path test_dataset.fasta --save_data_path test_data_features.h5 `

    Step 2: Perform prediction

    - `python src/virus_identification.py --fasta_file test_dataset.fasta --test_data_path test_data_features.h5 --result prediction.csv`


### phage_lifestyle_classification


    Step 1: Preprocess your data

    - `python src/pre-process.py --data_path test_dataset.fasta --save_data_path test_data_features.h5`

    Step 2: Perform prediction

    - `python src/phage_lifestyle_classification.py --fasta_file test_dataset.fasta --test_data_path test_data_features.h5 --result prediction.csv`


### plasmid_identification

    Step 1: Preprocess your data

    - `python src/pre-process.py --data_path test_dataset.fasta --save_data_path test_data_features.h5 `

    Step 2: Perform prediction

    - `python src/plasmid_identification.py --fasta_file test_dataset.fasta --test_data_path test_data_features.h5 --result prediction.csv`

## Using MetaGE for fine-tuning

    Step 1: Preprocess your data

    - `python src/pre-process.py --data_path dataset_label0.fasta --save_data_path dataset_label0_features.h5 --label 0`

    - `python src/pre-process.py --data_path dataset_label1.fasta --save_data_path dataset_label1_features.h5 --label 1`

    Merging dataset_label0_features.h5 and dataset_label1_features.h5 into Fine-tuning_features.h5


    Step 2: Perform fine-tuning

    - `python src/fine-tune.py --finetune pretrain_model.pth --train_data_path Fine-tuning_features.h5 test_data_path Fine-tuning_features_test.h5 --nb_classes 2`

## Pre-training MetaGE on your dataset

    Step 1: Preprocess your data

    - `python src/pre-process.py --data_path pretrain_dataset.fasta --save_data_path pretrain_data_features.h5 `

    Step 2: Perform pre-training

    - `python src/pre-train.py --data_path pretrain_data_features.h5`

## Model Data

Download it from the following link: https://drive.google.com/file/d/1QKzG-ol2iPo4_c8QazxnlfqqwJVgKtqL/view?usp=drive_link


## License

**License: All Rights Reserved (temporary)**  
After the related paper is published, this project will be released under the **MIT License** (or your chosen license).
