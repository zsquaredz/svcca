#!/bin/bash

source ~/.bashrc
conda activate cdt
cd /disk/ocean/zheng/svcca/


python analysis.py --data_dir1 ~/rds/hpc-work/summarization_svcca/out/activations/amazon_reviews/seed1/top5/epoch151/Books_last_hidden_state.npy --data_dir2 ~/rds/hpc-work/summarization_svcca/out/activations/amazon_reviews/seed1/Books/epoch151/Books_last_hidden_state.npy
