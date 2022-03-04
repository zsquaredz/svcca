#!/bin/bash

source /disk/ocean/zheng/.bashrc
conda activate cdt
cd /disk/ocean/zheng/svcca/

MODEL_CAT1=top5
DATA_CATEGORY1=Books
seed1=1

MODEL_CAT2=Books
DATA_CATEGORY2=Books
seed2=1

#layer=8
for layer in {0..11}
do
  epoch=0
  for head in {1..12}
  do
    echo "currently doing epoch ${epoch}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}-head-${head}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}-head-${head}"
    python analysis.py \
      --data_dir1 /disk/ocean/zheng/summarization_svcca/out/attentions/amazon_reviews/seed${seed1}/${MODEL_CAT1}/epoch${epoch}/${DATA_CATEGORY1}_layer_${layer}_max_attention_head${head}.npy \
      --data_dir2 /disk/ocean/zheng/summarization_svcca/out/attentions/amazon_reviews/seed${seed2}/${MODEL_CAT2}/epoch${epoch}/${DATA_CATEGORY2}_layer_${layer}_max_attention_head${head}.npy
  done

  for epoch in {1..202..10}
  do
    for for head in {1..12}
    do
      echo "currently doing epoch ${epoch}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}-head-${head}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}-head-${head}"
      python analysis.py \
        --data_dir1 /disk/ocean/zheng/summarization_svcca/out/attentions/amazon_reviews/seed${seed1}/${MODEL_CAT1}/epoch${epoch}/${DATA_CATEGORY1}_layer_${layer}_max_attention_head${head}.npy \
        --data_dir2 /disk/ocean/zheng/summarization_svcca/out/attentions/amazon_reviews/seed${seed2}/${MODEL_CAT2}/epoch${epoch}/${DATA_CATEGORY2}_layer_${layer}_max_attention_head${head}.npy
    done
  done
done
