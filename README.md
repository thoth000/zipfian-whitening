# Zipfian Whitening

This repository contains the code for the NeurIPS 2024 paper [Zipfian Whitening](https://openreview.net/forum?id=pASJxzMJb7) by Sho Yokoi, Han Bao, Hiroto Kurita and Hidetoshi Shimodaira.

## Overview
This repository mainly consists of two parts: experiments for static word embeddings and transformer-based embeddings.
- Experiments for static word embeddings: STS evaluation based on MTEB benchmark.
- Experiments for transformer-based embeddings: STS evaluation based on SimCSE's SentEval implementation. All the implementations are under the `SimCSE` directory, so please refer to the `SimCSE` directory for the details.
```
.
├── data # dataset files for static word embeddings experiments
├── notebooks # notebooks for visualization for static word embeddings experiments
├── results # results for static word embeddings experiments
├── scripts # scripts for running experiments for static word embeddings
├── src # source code for static word embeddings experiments
├── SimCSE # source code for transformer-based embeddings experiments
    ├── ...
├── ...
```

## Experiments for static word embeddings

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 2. Download pre-trained embeddings / convert to torch format
#### Download word2vec binary
- First, download word2vec binary to `models` dir from google drive https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
- Then run `bash scripts/prepare_embeddings.sh`
    - this handles download models other than word2vec and convert them to torch format
    - make sure you have `GoogleNews-vectors-negative300.bin.gz` in `models` dir before running this script


### 3. Reproduce the experimental results
```bash
source .venv/bin/activate
bash scripts/run.sh
```

### 4. Visualize the results
- Please refer to the notebooks in the `notebooks` directory for the visualization of the results.

### Note
- We used Enlgish Wikipedia word frequency from Arora+'17[1]: https://github.com/PrincetonML/SIF/raw/master/auxiliary_data/enwiki_vocab_min200.txt .
- For the calculation of IsoScore in `src/IsoScore.py`, we used the original implementation from Rudman+'22 [2]: https://github.com/bcbi-edu/p_eickhoff_isoscore/blob/main/IsoScore/IsoScore.py .

[1] Sanjeev Arora, Yingyu Liang and Tengyu Ma. "A Simple but Tough-to-Beat Baseline for Sentence Embeddings" In ICLR, 2017.  
[2] William Rudman, Nate Gillman, Taylor Rayne, and Carsten Eickhoff. "IsoScore: Measuring the Uniformity of Embedding Space Utilization" In Findings of ACL, 2022.

## Experiments for transformer-based embeddings

### 1. Install dependencies
Please go to the `SimCSE` directory and run the following command:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download STS datasets

Please go to the `SimCSE` directory and run the following command:
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

### 3. Reproduce the experimental results
Please go to the `SimCSE` directory and run the following command:
```bash
source .venv/bin/activate
bash scripts/run.sh
```

## Citation
If you use this codebase or find this work helpful, please cite:
```
@inproceedings{
yokoi2024zipfian,
    title={Zipfian Whitening},
    author={Sho Yokoi and Han Bao and Hiroto Kurita and Hidetoshi Shimodaira},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=pASJxzMJb7}
}
```