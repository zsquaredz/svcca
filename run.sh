#!/bin/bash

source /disk/ocean/zheng/.bashrc
conda activate cdt
cd /disk/ocean/zheng/svcca/



# for cate in 'Books' 'Clothing_Shoes_and_Jewelry' 'Electronics' 'Home_and_Kitchen' 'Movies_and_TV' 'Sports_and_Outdoors'

# m=10
# d=10  
for m in {10,50,100}
do
  for d in {10,50,100,200}
  do
    EXP_NAME1=${m}_model_${d}_data
    EXP_NAME2=reference_model

    if (( $m == 10 )) ; then
      SVD_DIM=70
      epoch1=501
    elif (( $m == 50 )) ; then
      SVD_DIM=350
      if (( $d == 10 )) ; then
        epoch1=501
      elif (( $d == 50 )) ; then
        epoch1=501
      elif (( $d == 100 )) ; then
        epoch1=501
      else
        epoch1=501
      fi
    else
      SVD_DIM=720
      if (( $d == 10 )) ; then
        epoch1=251
      elif (( $d == 50 )) ; then
        epoch1=151
      elif (( $d == 100 )) ; then
        epoch1=131
      else
        epoch1=131
      fi
    fi

    MODEL_CAT1=top5
    DATA_CATEGORY1=Books
    seed1=1

    MODEL_CAT2=Books
    DATA_CATEGORY2=Books
    seed2=1
    epoch2=201

    for layer in {0,12}
    do
      echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
      python analysis.py \
        --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
        --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
        --do_svcca \
        --svd_dim $SVD_DIM
    done
      
    MODEL_CAT1=top5
    DATA_CATEGORY1=Clothing_Shoes_and_Jewelry
    seed1=1

    MODEL_CAT2=Clothing_Shoes_and_Jewelry
    DATA_CATEGORY2=Clothing_Shoes_and_Jewelry
    seed2=1
    epoch2=201

    for layer in {0,12}
    do
      echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
      python analysis.py \
        --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
        --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
        --do_svcca \
        --svd_dim $SVD_DIM
    done

    MODEL_CAT1=top5
    DATA_CATEGORY1=Electronics
    seed1=1

    MODEL_CAT2=Electronics
    DATA_CATEGORY2=Electronics
    seed2=1
    epoch2=201

    for layer in {0,12}
    do
      echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
      python analysis.py \
        --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
        --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
        --do_svcca \
        --svd_dim $SVD_DIM
    done

    MODEL_CAT1=top5
    DATA_CATEGORY1=Home_and_Kitchen
    seed1=1

    MODEL_CAT2=Home_and_Kitchen
    DATA_CATEGORY2=Home_and_Kitchen
    seed2=1
    epoch2=201

    for layer in {0,12}
    do
      echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
      python analysis.py \
        --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
        --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
        --do_svcca \
        --svd_dim $SVD_DIM
    done

    MODEL_CAT1=top5
    DATA_CATEGORY1=Movies_and_TV
    seed1=1

    MODEL_CAT2=Movies_and_TV
    DATA_CATEGORY2=Movies_and_TV
    seed2=1
    epoch2=201

    for layer in {0,12}
    do
      echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
      python analysis.py \
        --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
        --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
        --do_svcca \
        --svd_dim $SVD_DIM
    done
  done
done

   


