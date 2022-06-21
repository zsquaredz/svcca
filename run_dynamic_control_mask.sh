#!/bin/bash
# this script will do the train dynamic calc as epoch progresses

source /disk/ocean/zheng/.bashrc
conda activate cdt
cd /disk/ocean/zheng/svcca/

# m=100 # 10, 25, 50, 75, 100
# SVD_DIM=700 # 68, 180, 365, 535, 700
mask_type=infreq50 # can be either general or specific for the mask file freq001/infreq50
for m in {10,25,50,75,100}
do
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
    if (( $m == 10 )) ; then
      SVD_DIM=68
      if (( $d == 10 )) ; then
        epoch1_end=501
        epoch2_best=501
      elif (( $d == 50 )) ; then
        epoch1_end=501
        epoch2_best=501
      elif (( $d == 100 )) ; then 
        epoch1_end=501
        epoch2_best=501
      else
        epoch1_end=501
        epoch2_best=501
      fi
    elif (( $m == 25 )) ; then
      SVD_DIM=180
      if (( $d == 10 )) ; then
        epoch1_end=501
        epoch2_best=501
      elif (( $d == 50 )) ; then
        epoch1_end=501
        epoch2_best=501
      elif (( $d == 100 )) ; then 
        epoch1_end=501
        epoch2_best=501
      else
        epoch1_end=501
        epoch2_best=501
      fi
    elif (( $m == 50 )) ; then
      SVD_DIM=365
      if (( $d == 10 )) ; then
        epoch1_end=501
        epoch2_best=501
      elif (( $d == 50 )) ; then
        epoch1_end=501
        epoch2_best=501
      elif (( $d == 100 )) ; then 
        epoch1_end=501
        epoch2_best=501
      else
        epoch1_end=501
        epoch2_best=501
      fi
    elif (( $m == 75 )) ; then
      SVD_DIM=535
      if (( $d == 10 )) ; then
        epoch1_end=501
        epoch2_best=181
      elif (( $d == 50 )) ; then
        epoch1_end=201
        epoch2_best=361
      elif (( $d == 100 )) ; then 
        epoch1_end=201
        epoch2_best=251
      else
        epoch1_end=201
        epoch2_best=211
      fi
    else
      SVD_DIM=700
      if (( $d == 10 )) ; then
        epoch1_end=251
        epoch2_best=101
      elif (( $d == 50 )) ; then
        epoch1_end=151
        epoch2_best=231
      elif (( $d == 100 )) ; then 
        epoch1_end=131
        epoch2_best=181
      else
        epoch1_end=131
        epoch2_best=151
      fi
    fi

    #layer=8
    for layer in {0,12}
    do
      epoch1=0
      epoch2=${epoch2_best}
      echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
      python analysis.py \
        --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
        --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
        --do_svcca \
        --use_mask \
        --mask_dir /disk/ocean/zheng/summarization_svcca/data/AmazonReviews/${DATA_CATEGORY1}/Test_2500_${DATA_CATEGORY1}.txt.${mask_type} \
        --svd_dim1 $SVD_DIM \
        --svd_dim2 $SVD_DIM

      for ((epoch=1;epoch<=${epoch1_end};epoch+=10))
      do
        epoch1=${epoch}
        epoch2=${epoch2_best}
        echo "currently doing ${EXP_NAME1} epoch ${epoch1}, ${EXP_NAME2} epoch ${epoch2}, seed-${seed1}-Model-${MODEL_CAT1}-layer-${layer}, and seed-${seed2}-Model-${MODEL_CAT2}-layer-${layer}"
        python analysis.py \
          --data_dir1 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed1}/${EXP_NAME1}/${MODEL_CAT1}/epoch${epoch1}/${DATA_CATEGORY1}_layer_${layer}_hidden_state.npy \
          --data_dir2 /disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed${seed2}/${EXP_NAME2}/${MODEL_CAT2}/epoch${epoch2}/${DATA_CATEGORY2}_layer_${layer}_hidden_state.npy \
          --do_svcca \
          --use_mask \
          --mask_dir /disk/ocean/zheng/summarization_svcca/data/AmazonReviews/${DATA_CATEGORY1}/Test_2500_${DATA_CATEGORY1}.txt.${mask_type} \
          --svd_dim1 $SVD_DIM \
          --svd_dim2 $SVD_DIM
      done
    done
  done
done
