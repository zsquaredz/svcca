from json.tool import main
import os, sys
os.environ['HOME'] = '/disk/ocean/zheng/' # for server only
os.environ['MPLCONFIGDIR'] = "/disk/ocean/zheng/.config/matplotlib/" # for server only
from matplotlib import pyplot as plt
# matplotlib inline
import numpy as np
import pickle
import pandas
import gzip
import argparse
import cca_core

def SVCCA(file1, file2, dim1_to_keep, dim2_to_keep, mask_file, use_mask=False):
    acts1 = np.load(file1) # data points x number of hidden dimension 
    acts2 = np.load(file2)
    # if use_mask:
    #     with open(mask_file) as f:
    #         word_mask_list = []
    #         for line in f.readlines():
    #             word_mask_list += [int(x) for x in line.strip().split()]
    #         word_mask = np.array(word_mask_list, dtype=bool)
    #         assert len(word_mask) == acts1.shape[0] # sanity check
    #         assert len(word_mask) == acts2.shape[0] # sanity check
    #         acts1 = acts1[word_mask]
    #         acts2 = acts2[word_mask]
    acts1 = np.float32(acts1)
    acts2 = np.float32(acts2)
    # print('file loaded')
    acts1 = acts1.T
    acts2 = acts2.T
    # print(acts1.shape) # need to be (number of neurons, number of test data points)
    # print(acts2.shape)

    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    
    print('starting to perform SVD')
    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False) # s1: min(row, col) V1 min(row, col) x data points size 
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    s1_sq = [i*i for i in s1]
    s2_sq = [i*i for i in s2]

    # print("Fraction of variance explained by 20 singular vectors", np.sum(s1[:20])/np.sum(s1))
    # print("Fraction of variance explained by 50 singular vectors", np.sum(s1[:50])/np.sum(s1))
    # print("Fraction of variance explained by 100 singular vectors", np.sum(s1[:100])/np.sum(s1))
    # print("Fraction of variance explained by 200 singular vectors", np.sum(s1[:200])/np.sum(s1))
    # print("Fraction of variance explained by 500 singular vectors", np.sum(s1[:500])/np.sum(s1))
    # print("Fraction of variance explained by 600 singular vectors", np.sum(s1[:600])/np.sum(s1))
    # print("Fraction of variance explained by 700 singular vectors", np.sum(s1[:700])/np.sum(s1))
    # print("Fraction of variance explained by 730 singular vectors", np.sum(s1[:730])/np.sum(s1))
    print("Fraction of variance explained by", dim1_to_keep ,"singular vectors for input1", np.sum(s1_sq[:dim1_to_keep])/np.sum(s1_sq))
    print("Fraction of variance explained by", dim2_to_keep ,"singular vectors for input2", np.sum(s2_sq[:dim2_to_keep])/np.sum(s2_sq))
    # print("Fraction of variance explained by 760 singular vectors", np.sum(s1[:760])/np.sum(s1))


    # print("Fraction of variance explained by", 345 ,"singular vectors for input1", np.sum(s1_sq[:345])/np.sum(s1_sq))
    # print("Fraction of variance explained by", 350 ,"singular vectors for input1", np.sum(s1_sq[:350])/np.sum(s1_sq))
    # print("Fraction of variance explained by", 355 ,"singular vectors for input1", np.sum(s1_sq[:355])/np.sum(s1_sq))
    # print("Fraction of variance explained by", 360 ,"singular vectors for input1", np.sum(s1_sq[:360])/np.sum(s1_sq))
    # print("Fraction of variance explained by", 365 ,"singular vectors for input1", np.sum(s1_sq[:365])/np.sum(s1_sq))
    # print("Fraction of variance explained by", 370 ,"singular vectors for input1", np.sum(s1_sq[:370])/np.sum(s1_sq))
    # print("Fraction of variance explained by", 685 ,"singular vectors for input2", np.sum(s2_sq[:685])/np.sum(s2_sq))
    # print("Fraction of variance explained by", 690 ,"singular vectors for input2", np.sum(s2_sq[:690])/np.sum(s2_sq))
    # print("Fraction of variance explained by", 695 ,"singular vectors for input2", np.sum(s2_sq[:695])/np.sum(s2_sq))
    # print("Fraction of variance explained by", 700 ,"singular vectors for input2", np.sum(s2_sq[:700])/np.sum(s2_sq))
    # print("Fraction of variance explained by", 710 ,"singular vectors for input2", np.sum(s2_sq[:710])/np.sum(s2_sq))
    # print("Fraction of variance explained by", 720 ,"singular vectors for input2", np.sum(s2_sq[:720])/np.sum(s2_sq))
    # print("Fraction of variance explained by", 730 ,"singular vectors for input2", np.sum(s2_sq[:730])/np.sum(s2_sq))
    # print("Fraction of variance explained by", 740 ,"singular vectors for input2", np.sum(s2_sq[:740])/np.sum(s2_sq))
    
    svacts1 = np.dot(s1[:dim1_to_keep]*np.eye(dim1_to_keep), V1[:dim1_to_keep]) # s1[:20]*np.eye(20) 20 x 20      V1[:20]   20 x number of data points  --> 20 x number of genreal tokens
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    # this will become dim1_to_keep x number of data points 
    svacts2 = np.dot(s2[:dim2_to_keep]*np.eye(dim2_to_keep), V2[:dim2_to_keep])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

    if use_mask:
        with open(mask_file) as f:
            word_mask_list = []
            for line in f.readlines():
                word_mask_list += [int(x) for x in line.strip().split()]
            word_mask = np.array(word_mask_list, dtype=bool)
            assert len(word_mask) == svacts1.shape[1] # sanity check
            assert len(word_mask) == svacts2.shape[1] # sanity check
            svacts1 = svacts1[:,word_mask]
            svacts2 = svacts2[:,word_mask]
    if use_mask:
        print(mask_file.split('.')[-1], 'mask has been applied, SVD done')
    else:
        print('SVD done')

    # svacts1, svacts2 = acts1, acts2

    # print('starting to perform CCA')
    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    print("result", np.mean(svcca_results["cca_coef1"]))

    # svacts1 = np.dot(s1[:200]*np.eye(200), V1[:200])
    # # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    # svacts2 = np.dot(s2[:200]*np.eye(200), V2[:200])
    # svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    # print("result for 20", np.mean(svcca_results["cca_coef1"]))

    # svacts1 = np.dot(s1[:500]*np.eye(500), V1[:500])
    # # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    # svacts2 = np.dot(s2[:500]*np.eye(500), V2[:500])
    # svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    # print("result for 20", np.mean(svcca_results["cca_coef1"]))

    # svacts1 = np.dot(s1[:750]*np.eye(750), V1[:750])
    # # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    # svacts2 = np.dot(s2[:750]*np.eye(750), V2[:750])
    # svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    # print("result for 20", np.mean(svcca_results["cca_coef1"]))

    # plt.plot(svcca_results["cca_coef1"], lw=2.0, label="MNIST")
    # plt.xlabel("Sorted CCA Correlation Coeff Idx")
    # plt.ylabel("CCA Correlation Coefficient Value")
    # plt.legend(loc="best")
    # plt.grid()

def Corr(file1, file2):
    attentions1 = np.load(file1)
    attentions2 = np.load(file2)
    corr = np.corrcoef(attentions1, attentions2)
    print("Corr result", corr[0][1])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument("--category1", type=str, default='Books', help="category")
    parser.add_argument("--data_dir1", type=str, default='./data/', help="Directory of data")
    parser.add_argument("--category2", type=str, default='Books', help="category")
    parser.add_argument("--data_dir2", type=str, default='./data/', help="Directory of data")
    parser.add_argument("--do_svcca", action='store_true', help="Whether to do SVCCA")
    parser.add_argument("--do_corr", action='store_true', help="Whether to do correlation")
    parser.add_argument("--use_mask", action='store_true', help="Whether to use the provided mask to apply to the data")
    parser.add_argument("--mask_dir", type=str, default='./data/', help="Directory of mask")
    parser.add_argument("--svd_dim1", type=int, default=750, help="Dimensions to keep after SVD")
    parser.add_argument("--svd_dim2", type=int, default=750, help="Dimensions to keep after SVD")
    args = parser.parse_args()
    
    if args.do_svcca:
        SVCCA(args.data_dir1, args.data_dir2, args.svd_dim1, args.svd_dim2, args.mask_dir, args.use_mask)
    elif args.do_corr:
        Corr(args.data_dir1, args.data_dir2)
    

    