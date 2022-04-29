#!/bin/bash

source /disk/ocean/zheng/.bashrc
conda activate cdt
cd /disk/ocean/zheng/svcca/



# for cate in 'Books' 'Clothing_Shoes_and_Jewelry' 'Electronics' 'Home_and_Kitchen' 'Movies_and_TV' 'Sports_and_Outdoors'

# m=10
# SVD_DIM=65
# d=10  

for m in {10,50,100}
do
  for d in {10,50,100,200}
  do
    EXP_NAME1=${m}_model_${d}_data
    EXP_NAME2=oracle
    if (( $m == 10 )) ; then
      SVD_DIM=68
      epoch1=501
    elif (( $m == 50 )) ; then
      SVD_DIM=365
      if (( $d == 10 )) ; then
        epoch1=501
      elif (( $d == 50 )) ; then
        epoch1=201
      elif (( $d == 100 )) ; then
        epoch1=101
      else
        epoch1=101
      fi
    else
      SVD_DIM=700
      if (( $d == 10 )) ; then
        epoch1=71
      elif (( $d == 50 )) ; then
        epoch1=51
      elif (( $d == 100 )) ; then
        epoch1=31
      else
        epoch1=31
      fi
    fi
    
    MODEL_CAT1=Sports_and_Outdoors
    DATA_CATEGORY1=Sports_and_Outdoors
    seed1=1
    
    MODEL_CAT2=Sports_and_Outdoors
    DATA_CATEGORY2=Sports_and_Outdoors
    seed2=1
    epoch2=81

    for layer in {0,12}
    do
      echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
      python analysis.py \
        --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
        --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
        --do_svcca \
        --svd_dim1 $SVD_DIM \
        --svd_dim2 700
    done
  done
done

   


