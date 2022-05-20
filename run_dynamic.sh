#!/bin/bash
# this script will do the train dynamic calc as epoch progresses

source /disk/ocean/zheng/.bashrc
conda activate cdt
cd /disk/ocean/zheng/svcca/


EXP_NAME1=100_model_200_data
EXP_NAME2=oracle

MODEL_CAT1=top5
DATA_CATEGORY1=Home_and_Kitchen
seed1=1

MODEL_CAT2=Home_and_Kitchen
DATA_CATEGORY2=Home_and_Kitchen
seed2=1
epoch2=171

SVD_DIM=700


#layer=8
for layer in {0,12}
do
  epoch1=0
  # epoch2=0
  echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
  python analysis.py \
    --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
    --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
    --do_svcca \
    --svd_dim1 $SVD_DIM \
    --svd_dim2 $SVD_DIM

  for epoch in {1..132..10}
  do
    epoch1=${epoch}
    echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
    python analysis.py \
      --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
      --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
      --do_svcca \
      --svd_dim1 $SVD_DIM \
      --svd_dim2 $SVD_DIM
  done
done
