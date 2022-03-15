#!/bin/bash

source /disk/ocean/zheng/.bashrc
conda activate cdt
cd /disk/ocean/zheng/svcca/


EXP_NAME=200_data

MODEL_CAT1=top5
DATA_CATEGORY1=Books
seed1=1

MODEL_CAT2=Books
DATA_CATEGORY2=Books
seed2=1

#layer=8
for layer in {0..12}
do
  epoch=0
  echo "currently doing epoch ${epoch}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
  python analysis.py \
    --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME}/${MODEL_CAT1}/epoch${epoch}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
    --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME}/${MODEL_CAT2}/epoch${epoch}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
    --do_svcca

  for epoch in {1..202..10}
  do
    echo "currently doing epoch ${epoch}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
    python analysis.py \
      --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME}/${MODEL_CAT1}/epoch${epoch}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
      --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME}/${MODEL_CAT2}/epoch${epoch}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
      --do_svcca
  done
done
