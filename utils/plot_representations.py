import json
import matplotlib
from transformers import AutoModelForMaskedLM
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import StandardScaler


def plot_embedding_weights():
    plt.figure(dpi=600)
    exp_names1 = ['10_model_100_data','25_model_100_data','50_model_100_data','75_model_100_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    epoch1s = ['501','501','501','201','131']
    epoch2s = ['501','501','501','251','181']
    for i in range(5):
        EXP_NAME1=exp_names1[i]
        MODEL_CAT1='top5'
        epoch1=epoch1s[i]

        EXP_NAME2=exp_names2[i]
        MODEL_CAT2='Books'
        epoch2=epoch2s[i]

        model_path1 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME1}/{MODEL_CAT1}-mlm/epoch{epoch1}'
        model_path2 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME2}/{MODEL_CAT2}-mlm/epoch{epoch2}'
        model1 = AutoModelForMaskedLM.from_pretrained(model_path1)
        embeddings_1 = model1.bert.embeddings.word_embeddings.weight.data
        embedding1_numpy = np.array(embeddings_1)
        model2 = AutoModelForMaskedLM.from_pretrained(model_path2)
        embeddings_2 = model2.bert.embeddings.word_embeddings.weight.data
        embedding2_numpy = np.array(embeddings_2)
        print(embedding1_numpy.shape)

        X = np.concatenate((embedding1_numpy,embedding2_numpy),axis=0)

        X_3d = PCA(n_components=2).fit_transform(X)
        print(X_3d.shape)
        data = {}
        data["general"] = X_3d[:30522]
        data["control"] = X_3d[30522:]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general', 'control'], ['3', (5,2)], ["blue", "red"]):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==4:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1]]
            h = [h[0], h[1]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=17.5,
                            framealpha=0.6,
                            markerscale=2)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.axis('off')
        fig.savefig("trial"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=1200,
                    transparent=True)

        plt.clf()

def plot_embedding_layer_representation():
    random.seed(30)
    indices = (random.sample(range(0,430923),k=2500))

    plt.figure(dpi=600)
    exp_names1 = ['10_model_100_data','25_model_100_data','50_model_100_data','75_model_100_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    epoch1s = ['501','501','501','201','131']
    epoch2s = ['501','501','501','251','181']
    for i in range(5):
        EXP_NAME1=exp_names1[i]
        MODEL_CAT1='top5'
        epoch1=epoch1s[i]

        EXP_NAME2=exp_names2[i]
        MODEL_CAT2='Books'
        epoch2=epoch2s[i]

        model_path1 = f'/disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed1/{EXP_NAME1}/{MODEL_CAT1}/epoch{epoch1}/Books_layer_0_hidden_state.npy'
        model_path2 = f'/disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed1/{EXP_NAME2}/{MODEL_CAT2}/epoch{epoch2}/Books_layer_0_hidden_state.npy'
        
        acts1 = np.load(model_path1) # data points x number of hidden dimension 
        acts2 = np.load(model_path2)
        print(acts1.shape)

        X = np.concatenate((acts1, acts2),axis=0)
        X = StandardScaler().fit_transform(X)
        X_3d = PCA(n_components=2).fit_transform(X)
        print(X_3d.shape)
        data = {}
        data["general"] = np.take(X_3d[:430923], indices, axis=0)
        data["control"] = np.take(X_3d[430923:], indices, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general', 'control'], ['3', (5,2)], ["blue", "red"]):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==4:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1]]
            h = [h[0], h[1]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=17.5,
                            framealpha=0.6,
                            markerscale=2)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.axis('off')
        fig.savefig("trial_embedding_layer_representation"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=600,
                    transparent=True)

        plt.clf()

def plot_final_layer_weights():
    plt.figure(dpi=600)
    exp_names1 = ['10_model_100_data','25_model_100_data','50_model_100_data','75_model_100_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    epoch1s = ['501','501','501','201','131']
    epoch2s = ['501','501','501','251','181']
    for i in range(5):
        EXP_NAME1=exp_names1[i]
        MODEL_CAT1='top5'
        epoch1=epoch1s[i]

        EXP_NAME2=exp_names2[i]
        MODEL_CAT2='Books'
        epoch2=epoch2s[i]

        model_path1 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME1}/{MODEL_CAT1}-mlm/epoch{epoch1}'
        model_path2 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME2}/{MODEL_CAT2}-mlm/epoch{epoch2}'
        model1 = AutoModelForMaskedLM.from_pretrained(model_path1)
        embeddings_1 = model1.bert.encoder.layer[11].output.dense.weight.data
        embedding1_numpy = np.array(embeddings_1)
        embedding1_numpy = embedding1_numpy.T

        model2 = AutoModelForMaskedLM.from_pretrained(model_path2)
        embeddings_2 = model2.bert.encoder.layer[11].output.dense.weight.data
        embedding2_numpy = np.array(embeddings_2)
        embedding2_numpy = embedding2_numpy.T
        print(embedding2_numpy.shape)

        X = np.concatenate((embedding1_numpy,embedding2_numpy),axis=0)

        X_3d = PCA(n_components=2).fit_transform(X)
        print(X_3d.shape)
        data = {}
        data["general"] = X_3d[:3072]
        data["control"] = X_3d[3072:]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general', 'control'], ['3', (5,2)], ["blue", "red"]):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==4:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1]]
            h = [h[0], h[1]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=17.5,
                            framealpha=0.6,
                            markerscale=2)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.axis('off')
        fig.savefig("trial_cls_decoder"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=1200,
                    transparent=True)

        plt.clf()

def plot_final_layer_representation():
    random.seed(30)
    indices = (random.sample(range(0,430923),k=2500))

    plt.figure(dpi=600)
    exp_names1 = ['10_model_100_data','25_model_100_data','50_model_100_data','75_model_100_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    epoch1s = ['501','501','501','201','131']
    epoch2s = ['501','501','501','251','181']
    for i in range(5):
        EXP_NAME1=exp_names1[i]
        MODEL_CAT1='top5'
        epoch1=epoch1s[i]

        EXP_NAME2=exp_names2[i]
        MODEL_CAT2='Books'
        epoch2=epoch2s[i]

        model_path1 = f'/disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed1/{EXP_NAME1}/{MODEL_CAT1}/epoch{epoch1}/Books_layer_12_hidden_state.npy'
        model_path2 = f'/disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed1/{EXP_NAME2}/{MODEL_CAT2}/epoch{epoch2}/Books_layer_12_hidden_state.npy'
        
        acts1 = np.load(model_path1) # data points x number of hidden dimension 
        acts2 = np.load(model_path2)
        print(acts1.shape)

        X = np.concatenate((acts1, acts2),axis=0)
        X = StandardScaler().fit_transform(X)
        X_3d = PCA(n_components=2).fit_transform(X)
        print(X_3d.shape)
        data = {}
        data["general"] = np.take(X_3d[:430923], indices, axis=0)
        data["control"] = np.take(X_3d[430923:], indices, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general', 'control'], ['3', (5,2)], ["blue", "red"]):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==4:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1]]
            h = [h[0], h[1]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=17.5,
                            framealpha=0.6,
                            markerscale=2)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.axis('off')
        fig.savefig("trial_final_layer_representation"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=600,
                    transparent=True)

        plt.clf()

if __name__ == '__main__':
    # plot_embedding_weights()
    plot_embedding_layer_representation()
    # plot_final_layer_weights()
    # plot_final_layer_representation()