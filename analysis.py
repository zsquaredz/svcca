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

def SVCCA(file1, file2):
    acts1 = np.load(file1)
    acts2 = np.load(file2)
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
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    # print("Fraction of variance explained by 20 singular vectors", np.sum(s1[:20])/np.sum(s1))
    # print("Fraction of variance explained by 50 singular vectors", np.sum(s1[:50])/np.sum(s1))
    # print("Fraction of variance explained by 100 singular vectors", np.sum(s1[:100])/np.sum(s1))
    # print("Fraction of variance explained by 200 singular vectors", np.sum(s1[:200])/np.sum(s1))
    # print("Fraction of variance explained by 500 singular vectors", np.sum(s1[:500])/np.sum(s1))
    # print("Fraction of variance explained by 600 singular vectors", np.sum(s1[:600])/np.sum(s1))
    # print("Fraction of variance explained by 700 singular vectors", np.sum(s1[:700])/np.sum(s1))
    # print("Fraction of variance explained by 730 singular vectors", np.sum(s1[:730])/np.sum(s1))
    print("Fraction of variance explained by 750 singular vectors", np.sum(s1[:750])/np.sum(s1))
    # print("Fraction of variance explained by 760 singular vectors", np.sum(s1[:760])/np.sum(s1))
    dim_to_keep = 750
    svacts1 = np.dot(s1[:dim_to_keep]*np.eye(dim_to_keep), V1[:dim_to_keep])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:dim_to_keep]*np.eye(dim_to_keep), V2[:dim_to_keep])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)
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
    args = parser.parse_args()
    
    # SVCCA(args.data_dir1, args.data_dir2)
    Corr(args.data_dir1, args.data_dir2)