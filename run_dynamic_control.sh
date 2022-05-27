#!/bin/bash
# this script will do the train dynamic calc as epoch progresses

source /disk/ocean/zheng/.bashrc
conda activate cdt
cd /disk/ocean/zheng/svcca/

m=100 # 10, 25, 50, 75, 100
SVD_DIM=700 # 68, 180, 365, 535, 700
for d in {10,50,100,200}
  do
  EXP_NAME1=${m}_model_${d}_data
  EXP_NAME2=new_control_${m}_model_${d}_data

  MODEL_CAT1=top5
  DATA_CATEGORY1=Books
  seed1=1

  MODEL_CAT2=Books
  DATA_CATEGORY2=Books
  seed2=1
  # epoch2=171
  if (( $d == 10 )) ; then
    if [ $cate == "Books" ]; then
      epoch_end=101
    elif [ $cate == "Clothing_Shoes_and_Jewelry" ]; then
      epoch_end=251111
    elif [ $cate == 'Electronics' ]; then
      epoch_end=251111
    elif [ $cate == 'Home_and_Kitchen' ];  then
      epoch_end=251111
    elif [ $cate == 'Movies_and_TV' ];  then
      epoch_end=251111
    else
      epoch_end=999999999 # we should never end up here
    fi
  elif (( $d == 50 )) ; then
    if [ $cate == 'Books' ]; then
      epoch_end=231
    elif [ $cate == 'Clothing_Shoes_and_Jewelry' ]; then
      epoch_end=151111
    elif [ $cate == 'Electronics' ]; then
      epoch_end=151111
    elif [ $cate == 'Home_and_Kitchen' ];  then
      epoch_end=151111
    elif [ $cate == 'Movies_and_TV' ];  then
      epoch_end=151111
    else
      epoch_end=999999999 # we should never end up here
    fi
  elif (( $d == 100 )) ; then
    if [ $cate == 'Books' ]; then
      epoch_end=181
    elif [ $cate == 'Clothing_Shoes_and_Jewelry' ]; then
      epoch_end=131111
    elif [ $cate == 'Electronics' ]; then
      epoch_end=131111
    elif [ $cate == 'Home_and_Kitchen' ];  then
      epoch_end=131111
    elif [ $cate == 'Movies_and_TV' ];  then
      epoch_end=131111
    else
      epoch_end=999999999 # we should never end up here
    fi
  else
    if [ $cate == 'Books' ]; then
      epoch_end=151
    elif [ $cate == 'Clothing_Shoes_and_Jewelry' ]; then
      epoch_end=131111
    elif [ $cate == 'Electronics' ]; then
      epoch_end=131111
    elif [ $cate == 'Home_and_Kitchen' ];  then
      epoch_end=131111
    elif [ $cate == 'Movies_and_TV' ];  then
      epoch_end=131111
    else
      epoch_end=999999999 # we should never end up here
    fi
  fi

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

    for ((epoch=1;epoch<=${epoch_end};epoch+=10))
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
done
