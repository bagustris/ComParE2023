***Forked from [EIHW/ComParE2023](http:github.com/EIHW/ComParE2023)***

# ComParE23 - The Hume-Prosody Corpus (HP-C)
This repository provides the code for running my participation for The Hume-Prosody Corpus (HP-C) subchallenge of ComParE2023 (excluding feature extraction).


## Getting the code
Clone this repository and checkout the correct branch:
```bash
git clone https://github.com/bagustris/ComParE2023
```

## Adding the data
Drop the data into `./data` (~40GB), creating this directory structure:
```console
data
├── features
│  ├── audeep
│  ├── deepspectrum
│  ├── opensmile
│  └── wav2vec
├── lab
├── raw
│  └── wav
└── wav
```
You can make a soft link (like in this repo) if your data is located elsewhere (e.g., in `/data/`).

```
ln -sf /data/14_ComParE23_HPC_AIST-SPRT/data ./data
```

## Creating Virtual Enviroments via Miniconda
Create virtual environment with python 3.9:

`conda create -n ComParE2023 python=3.9`

Install dependencies:  
`pip install -r requirements.txt`

Run the experiments:  
`python3 src/ml/svm.py wav2vec`

Calculate the results' score:  
`python3 src/ml/metrics.py wav2vec`


## Extracting features
To extract feature from Hugging Face, you can use `feat_extract.py` with 
arguments `name` [output directory] and `Hugging Face model name` [e.g. facebook/wav2vec2-large-xlsr-53].

Format:  
`./feat_extract.py [output directory] [Hugging Face model name] [device]`

Example:
```bash
./feat_extract.py xlsr-53 jonatasgrosman/wav2vec2-large-xlsr-53-english
``` 

You need to change permission (`chmod +x feat_extract.py`) to run the script directly.

