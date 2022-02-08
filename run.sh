#!/bin/bash

source ~/.bashrc
conda activate cdt
cd /disk/ocean/zheng/svcca/

MODEL_CAT1=top5
DATA_CATEGORY1=Books
seed1=1

MODEL_CAT2=Books
DATA_CATEGORY2=Books
seed2=1

layer=7

epoch=0
echo "currently doing epoch ${epoch}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
python analysis.py \
  --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${MODEL_CAT1}/epoch${epoch}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
  --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${MODEL_CAT2}/epoch${epoch}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy

for epoch in {1..202..10}
do
  echo "currently doing epoch ${epoch}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
  python analysis.py \
    --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${MODEL_CAT1}/epoch${epoch}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
    --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${MODEL_CAT2}/epoch${epoch}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy
done
