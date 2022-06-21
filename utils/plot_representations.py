import json
import matplotlib
from transformers import AutoModelForMaskedLM
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text


def plot_embedding_weights():
    plt.figure(dpi=600)
    exp_names1 = ['10_model_100_data','25_model_100_data','50_model_100_data','75_model_100_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    epoch1s = ['501','501','501','201','131']
    epoch2s = ['501','501','501','251','181'] # books
    # epoch2s = ['501','501','501','501','501'] # clothing
    for i in range(5):
        EXP_NAME1=exp_names1[i]
        MODEL_CAT1='top5'
        epoch1=epoch1s[i]

        EXP_NAME2=exp_names2[i]
        MODEL_CAT2='Books'
        # MODEL_CAT2='Clothing_Shoes_and_Jewelry'
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
        X = StandardScaler().fit_transform(X)
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

def plot_embedding_weights1():
    plt.figure(dpi=600)
    exp_names1 = ['10_model_10_data','10_model_100_data','100_model_10_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_10_data','new_control_10_model_100_data','new_control_100_model_10_data','new_control_100_model_100_data']
    epoch1s = ['501','501','251','131']
    epoch2s = ['501','501','101','181'] # books
    specific_words = [(7592, 'hello'), (2646, 'toward'), (7615, 'comment'), (4952, 'listen'), (3071, 'everyone')] 
    general_words = [(22524, 'appendix'), (8544,'publishers'), (8882, 'curriculum'), (24402, 'grammatical'), (18534, 'autobiographical')]
    for i in range(4):
        EXP_NAME1=exp_names1[i]
        MODEL_CAT1='top5'
        epoch1=epoch1s[i]

        EXP_NAME2=exp_names2[i]
        MODEL_CAT2='Books'
        # MODEL_CAT2='Clothing_Shoes_and_Jewelry'
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
        X = StandardScaler().fit_transform(X)
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
            # if label == 'general':
            #     texts=[ax.text(X_temp[idx,0], X_temp[idx,1], 'g*', fontsize=12.5, color='black') for idx,word in specific_words]
            #     texts=[ax.text(X_temp[idx,0], X_temp[idx,1], 'g', fontsize=12.5, color='black') for idx,word in general_words]
            # else:
            #     texts=[ax.text(X_temp[idx,0], X_temp[idx,1], 'c*', fontsize=12.5, color='black') for idx,word in specific_words]
            #     texts=[ax.text(X_temp[idx,0], X_temp[idx,1], 'c', fontsize=12.5, color='black') for idx,word in general_words]
            # adjust_text(texts)
        if i==3:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1]]
            h = [h[0], h[1]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=12,
                            framealpha=0.6,
                            markerscale=1)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # ax.axis('off')
        fig.savefig("trial-"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=1200,
                    transparent=True)

        plt.clf()

def plot_embedding_weights2():
    plt.figure(dpi=600)
    exp_names1 = ['10_model_10_data','10_model_100_data','100_model_10_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_10_data','new_control_10_model_100_data','new_control_100_model_10_data','new_control_100_model_100_data']
    epoch1s = ['501','501','251','131']
    epoch2s = ['501','501','101','181'] # books
    specific_words = [(7592, 'hello'), (2646, 'toward'), (7615, 'comment'), (4952, 'listen'), (3071, 'everyone')] 
    general_words = [(22524, 'appendix'), (8544,'publishers'), (8882, 'curriculum'), (24402, 'grammatical'), (18534, 'autobiographical')]
    for i in range(4):
        EXP_NAME1=exp_names1[i]
        MODEL_CAT1='top5'
        epoch1=epoch1s[i]

        EXP_NAME2=exp_names2[i]
        MODEL_CAT2='Books'
        # MODEL_CAT2='Clothing_Shoes_and_Jewelry'
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
        X = StandardScaler().fit_transform(X)
        X_3d = PCA(n_components=2).fit_transform(X)
        print(X_3d.shape)
        data = {}
        data["general"] = X_3d[:30522]
        data["control"] = X_3d[30522:]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # for label, marker, color in zip(['general'], ['3'], ["blue"]):
        for label, marker, color in zip(['control'], [(5,2)], ["red"]):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
            if label == 'general':
                texts=[ax.text(X_temp[idx,0], X_temp[idx,1], 's', fontsize=12.5, color='black') for idx,word in specific_words]
                texts=[ax.text(X_temp[idx,0], X_temp[idx,1], 'g', fontsize=12.5, color='black') for idx,word in general_words]
            else:
                texts=[ax.text(X_temp[idx,0], X_temp[idx,1], 's', fontsize=12.5, color='black') for idx,word in specific_words]
                texts=[ax.text(X_temp[idx,0], X_temp[idx,1], 'g', fontsize=12.5, color='black') for idx,word in general_words]
            adjust_text(texts)
        if i==3:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0]]
            h = [h[0]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=12,
                            framealpha=0.6,
                            markerscale=1)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # ax.axis('off')
        fig.savefig("trial-g-"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=1200,
                    transparent=True)

        plt.clf()

def plot_five_embedding_weights():
    plt.figure(dpi=600)
    exp_names1 = ['10_model_100_data','25_model_100_data','50_model_100_data','75_model_100_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    exp_names3 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    exp_names4 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    exp_names5 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    exp_names6 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    epoch1s = ['501','501','501','201','131'] # general
    epoch2s = ['501','501','501','251','181'] # books
    epoch3s = ['501','501','501','501','501'] # clothing
    epoch4s = ['501','501','501','251','151'] # electronics
    epoch5s = ['501','501','501','251','151'] # home
    epoch6s = ['501','501','501','251','181'] # movie
    for i in range(5):
        EXP_NAME1=exp_names1[i]
        MODEL_CAT1='top5'
        epoch1=epoch1s[i]

        EXP_NAME2=exp_names2[i]
        MODEL_CAT2='Books'
        epoch2=epoch2s[i]

        EXP_NAME3=exp_names3[i]
        MODEL_CAT3='Clothing_Shoes_and_Jewelry'
        epoch3=epoch3s[i]

        EXP_NAME4=exp_names4[i]
        MODEL_CAT4='Electronics'
        epoch4=epoch4s[i]

        EXP_NAME5=exp_names5[i]
        MODEL_CAT5='Home_and_Kitchen'
        epoch5=epoch5s[i]

        EXP_NAME6=exp_names6[i]
        MODEL_CAT6='Movies_and_TV'
        epoch6=epoch6s[i]

        model_path1 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME1}/{MODEL_CAT1}-mlm/epoch{epoch1}'
        model_path2 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME2}/{MODEL_CAT2}-mlm/epoch{epoch2}'
        model_path3 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME3}/{MODEL_CAT3}-mlm/epoch{epoch3}'
        model_path4 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME4}/{MODEL_CAT4}-mlm/epoch{epoch4}'
        model_path5 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME5}/{MODEL_CAT5}-mlm/epoch{epoch5}'
        model_path6 = f'/disk/ocean/zheng/summarization_svcca/checkpoints/bert_base_uncased/amazon_reviews/seed1/{EXP_NAME6}/{MODEL_CAT6}-mlm/epoch{epoch6}'
        model1 = AutoModelForMaskedLM.from_pretrained(model_path1)
        embeddings_1 = model1.bert.embeddings.word_embeddings.weight.data
        embedding1_numpy = np.array(embeddings_1)
        model2 = AutoModelForMaskedLM.from_pretrained(model_path2)
        embeddings_2 = model2.bert.embeddings.word_embeddings.weight.data
        embedding2_numpy = np.array(embeddings_2)
        model3 = AutoModelForMaskedLM.from_pretrained(model_path3)
        embeddings_3 = model3.bert.embeddings.word_embeddings.weight.data
        embedding3_numpy = np.array(embeddings_3)
        model4 = AutoModelForMaskedLM.from_pretrained(model_path4)
        embeddings_4 = model4.bert.embeddings.word_embeddings.weight.data
        embedding4_numpy = np.array(embeddings_4)
        model5 = AutoModelForMaskedLM.from_pretrained(model_path5)
        embeddings_5 = model5.bert.embeddings.word_embeddings.weight.data
        embedding5_numpy = np.array(embeddings_5)
        model6 = AutoModelForMaskedLM.from_pretrained(model_path6)
        embeddings_6 = model6.bert.embeddings.word_embeddings.weight.data
        embedding6_numpy = np.array(embeddings_6)
        print(embedding1_numpy.shape)

        X = np.concatenate((embedding1_numpy,embedding2_numpy,embedding3_numpy,embedding4_numpy,embedding5_numpy,embedding6_numpy),axis=0)
        X = StandardScaler().fit_transform(X)
        X_3d = PCA(n_components=2).fit_transform(X)
        print(X_3d.shape)
        data = {}
        data["general"] = X_3d[:30522]
        data["book"] = X_3d[30522:30522*2]
        data["clothing"] = X_3d[30522*2:30522*3]
        data["electronics"] = X_3d[30522*3:30522*4]
        data["home"] = X_3d[30522*4:30522*5]
        data["movie"] = X_3d[30522*5:30522*6]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general', 'book', 'clothing', 'electronics', 'home', 'movie'], ['3', (5,2), '+', 'x', '1', '2'], ["blue", "red", 'green', 'cyan', 'yellow', 'magenta']):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==4:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1], l[2], l[3], l[4], l[5]]
            h = [h[0], h[1], h[2], h[3], h[4], h[5]]
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
        fig.savefig("trial_all"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=1200,
                    transparent=True)

        plt.clf()

def plot_embedding_layer_representation():
    random.seed(30)
    indices = (random.sample(range(0,430923),k=2500)) # books: 430923 clothing: 117499

    plt.figure(dpi=600)
    exp_names1 = ['10_model_100_data','25_model_100_data','50_model_100_data','75_model_100_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    epoch1s = ['501','501','501','201','131']
    epoch2s = ['501','501','501','251','181'] # books
    # epoch2s = ['501','501','501','501','501'] # clothing
    for i in range(5):
        EXP_NAME1=exp_names1[i]
        MODEL_CAT1='top5'
        epoch1=epoch1s[i]

        EXP_NAME2=exp_names2[i]
        MODEL_CAT2='Books' # Clothing_Shoes_and_Jewelry
        epoch2=epoch2s[i]

        model_path1 = f'/disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed1/{EXP_NAME1}/{MODEL_CAT1}/epoch{epoch1}/Books_layer_0_hidden_state.npy'
        model_path2 = f'/disk/ocean/zheng/summarization_svcca/out/activations/amazon_reviews/seed1/{EXP_NAME2}/{MODEL_CAT2}/epoch{epoch2}/Books_layer_0_hidden_state.npy'
        
        acts1 = np.load(model_path1) # data points x number of hidden dimension 
        acts2 = np.load(model_path2)
        print(acts1.shape)

        with open(general_mask_file) as f:
            word_mask_list = []
            for line in f.readlines():
                word_mask_list += [int(x) for x in line.strip().split()]
            word_mask = np.array(word_mask_list, dtype=bool)
            assert len(word_mask) == acts1.shape[1] # sanity check
            assert len(word_mask) == acts2.shape[1] # sanity check
            acts1 = acts1[:,word_mask]
            acts2 = acts2[:,word_mask]

        X = np.concatenate((acts1, acts2),axis=0)
        X = StandardScaler().fit_transform(X)
        X_3d = PCA(n_components=2).fit_transform(X)
        print(X_3d.shape)
        data = {}
        data["general"] = np.take(X_3d[:430923], indices, axis=0) # books: 430923 clothing: 117499
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

def plot_embedding_layer_representation_with_mask1():
    # random.seed(30)
    # indices = (random.sample(range(0,117499),k=2500)) # books: 430923 clothing: 117499

    plt.figure(dpi=600)
    exp_names1 = ['10_model_10_data','10_model_100_data','100_model_10_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_10_data','new_control_10_model_100_data','new_control_100_model_10_data','new_control_100_model_100_data']
    epoch1s = ['501','501','251','131']
    epoch2s = ['501','501','101','181'] # books
    # epoch2s = ['501','501','501','501','501'] # clothing
    for i in range(4):
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
        size = X_3d.shape[0]//2

        general_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.general'
        specific_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.specific'
        with open(general_mask_file) as f:
            word_mask_list_gen = []
            for line in f.readlines():
                word_mask_list_gen += [int(x) for x in line.strip().split()]
        word_mask_gen = np.array(word_mask_list_gen, dtype=bool)
        assert len(word_mask_gen) == acts1.shape[0] # sanity check

        with open(specific_mask_file) as f:
            word_mask_list_spe = []
            for line in f.readlines():
                word_mask_list_spe += [int(x) for x in line.strip().split()]
        word_mask_spe = np.array(word_mask_list_spe, dtype=bool)
        assert len(word_mask_spe) == acts1.shape[0] # sanity check

        general_data = X_3d[:size]
        general_data_gen = general_data[word_mask_gen]
        general_data_spe = general_data[word_mask_spe]

        control_data = X_3d[size:]
        control_data_gen = control_data[word_mask_gen]
        control_data_spe = control_data[word_mask_spe]
        
        random.seed(30)
        indices_gen = (random.sample(range(0,general_data_gen.shape[0]), k=1000))
        indices_spe = (random.sample(range(0,general_data_spe.shape[0]), k=1000))
        data = {}
        data["general-general"] = np.take(general_data_gen, indices_gen, axis=0) # books: 430923 clothing: 117499
        data["control-general"] = np.take(control_data_gen, indices_gen, axis=0)
        data["general-specific"] = np.take(general_data_spe, indices_spe, axis=0) # books: 430923 clothing: 117499
        data["control-specific"] = np.take(control_data_spe, indices_spe, axis=0)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general-general', 'control-general', 'general-specific', 'control-specific'], ['3', (5,2), '+', '1'], ["blue", 'red', 'cyan', 'magenta']):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==3:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1], l[2], l[3]]
            h = [h[0], h[1], h[2], h[3]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=12,
                            framealpha=0.6,
                            markerscale=1)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.axis('off')
        fig.savefig("trial_embedding_layer_representation_mask-"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=600,
                    transparent=True)

        plt.clf()

def plot_embedding_layer_representation_with_mask_model():
    # random.seed(30)
    # indices = (random.sample(range(0,117499),k=2500)) # books: 430923 clothing: 117499

    plt.figure(dpi=600)
    exp_names1 = ['10_model_100_data','25_model_100_data','50_model_100_data','75_model_100_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    epoch1s = ['501','501','501','201','131']
    epoch2s = ['501','501','501','251','181'] # books
    # epoch2s = ['501','501','501','501','501'] # clothing
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
        size = X_3d.shape[0]//2

        general_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.general'
        specific_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.specific'
        with open(general_mask_file) as f:
            word_mask_list_gen = []
            for line in f.readlines():
                word_mask_list_gen += [int(x) for x in line.strip().split()]
        word_mask_gen = np.array(word_mask_list_gen, dtype=bool)
        assert len(word_mask_gen) == acts1.shape[0] # sanity check

        with open(specific_mask_file) as f:
            word_mask_list_spe = []
            for line in f.readlines():
                word_mask_list_spe += [int(x) for x in line.strip().split()]
        word_mask_spe = np.array(word_mask_list_spe, dtype=bool)
        assert len(word_mask_spe) == acts1.shape[0] # sanity check

        general_data = X_3d[:size]
        general_data_gen = general_data[word_mask_gen]
        general_data_spe = general_data[word_mask_spe]

        control_data = X_3d[size:]
        control_data_gen = control_data[word_mask_gen]
        control_data_spe = control_data[word_mask_spe]
        
        random.seed(30)
        indices_gen = (random.sample(range(0,general_data_gen.shape[0]), k=1000))
        indices_spe = (random.sample(range(0,general_data_spe.shape[0]), k=1000))
        data = {}
        data["general-gen"] = np.take(general_data_gen, indices_gen, axis=0) # books: 430923 clothing: 117499
        data["control-gen"] = np.take(control_data_gen, indices_gen, axis=0)
        data["general-spe"] = np.take(general_data_spe, indices_spe, axis=0) # books: 430923 clothing: 117499
        data["control-spe"] = np.take(control_data_spe, indices_spe, axis=0)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general-gen', 'control-gen', 'general-spe', 'control-spe'], ['3', (5,2), '+', '1'], ["blue", 'red', 'cyan', 'magenta']):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==4:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1], l[2], l[3]]
            h = [h[0], h[1], h[2], h[3]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=12.5,
                            framealpha=0.6,
                            markerscale=1)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.axis('off')
        fig.savefig("trial_embedding_layer_representation_mask"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=600,
                    transparent=True)

        plt.clf()

def plot_embedding_layer_representation_with_mask_data():
    # random.seed(30)
    # indices = (random.sample(range(0,117499),k=2500)) # books: 430923 clothing: 117499

    plt.figure(dpi=600)
    exp_names1 = ['100_model_10_data','100_model_50_data','100_model_100_data','100_model_200_data']
    exp_names2 = ['new_control_100_model_10_data','new_control_100_model_50_data','new_control_100_model_100_data','new_control_100_model_200_data']
    epoch1s = ['251','151','131','131']
    epoch2s = ['101','231','181','151'] # books
    # epoch2s = ['501','501','501','501','501'] # clothing
    for i in range(4):
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
        size = X_3d.shape[0]//2

        general_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.general'
        specific_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.specific'
        with open(general_mask_file) as f:
            word_mask_list_gen = []
            for line in f.readlines():
                word_mask_list_gen += [int(x) for x in line.strip().split()]
        word_mask_gen = np.array(word_mask_list_gen, dtype=bool)
        assert len(word_mask_gen) == acts1.shape[0] # sanity check

        with open(specific_mask_file) as f:
            word_mask_list_spe = []
            for line in f.readlines():
                word_mask_list_spe += [int(x) for x in line.strip().split()]
        word_mask_spe = np.array(word_mask_list_spe, dtype=bool)
        assert len(word_mask_spe) == acts1.shape[0] # sanity check

        general_data = X_3d[:size]
        general_data_gen = general_data[word_mask_gen]
        general_data_spe = general_data[word_mask_spe]

        control_data = X_3d[size:]
        control_data_gen = control_data[word_mask_gen]
        control_data_spe = control_data[word_mask_spe]
        
        random.seed(30)
        indices_gen = (random.sample(range(0,general_data_gen.shape[0]), k=1000))
        indices_spe = (random.sample(range(0,general_data_spe.shape[0]), k=1000))
        data = {}
        data["general-gen"] = np.take(general_data_gen, indices_gen, axis=0) # books: 430923 clothing: 117499
        data["control-gen"] = np.take(control_data_gen, indices_gen, axis=0)
        data["general-spe"] = np.take(general_data_spe, indices_spe, axis=0) # books: 430923 clothing: 117499
        data["control-spe"] = np.take(control_data_spe, indices_spe, axis=0)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general-gen', 'control-gen', 'general-spe', 'control-spe'], ['3', (5,2), '+', '1'], ["blue", 'red', 'cyan', 'magenta']):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==4:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1], l[2], l[3]]
            h = [h[0], h[1], h[2], h[3]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=12.5,
                            framealpha=0.6,
                            markerscale=1)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.axis('off')
        fig.savefig("trial_embedding_layer_representation_mask_data"+str(i)+".pdf",
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
        # data["general"] = X_3d[:430923]
        # data["control"] = X_3d[430923:]

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

def plot_final_layer_representation_with_mask():
    # random.seed(30)
    # indices = (random.sample(range(0,117499),k=2500)) # books: 430923 clothing: 117499

    plt.figure(dpi=600)
    exp_names1 = ['10_model_100_data','25_model_100_data','50_model_100_data','75_model_100_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_100_data','new_control_25_model_100_data','new_control_50_model_100_data','new_control_75_model_100_data','new_control_100_model_100_data']
    epoch1s = ['501','501','501','201','131']
    epoch2s = ['501','501','501','251','181'] # books
    # epoch2s = ['501','501','501','501','501'] # clothing
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
        size = X_3d.shape[0]//2

        general_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.general'
        specific_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.specific'
        with open(general_mask_file) as f:
            word_mask_list_gen = []
            for line in f.readlines():
                word_mask_list_gen += [int(x) for x in line.strip().split()]
        word_mask_gen = np.array(word_mask_list_gen, dtype=bool)
        assert len(word_mask_gen) == acts1.shape[0] # sanity check

        with open(specific_mask_file) as f:
            word_mask_list_spe = []
            for line in f.readlines():
                word_mask_list_spe += [int(x) for x in line.strip().split()]
        word_mask_spe = np.array(word_mask_list_spe, dtype=bool)
        assert len(word_mask_spe) == acts1.shape[0] # sanity check

        general_data = X_3d[:size]
        general_data_gen = general_data[word_mask_gen]
        general_data_spe = general_data[word_mask_spe]

        control_data = X_3d[size:]
        control_data_gen = control_data[word_mask_gen]
        control_data_spe = control_data[word_mask_spe]
        
        random.seed(30)
        indices_gen = (random.sample(range(0,general_data_gen.shape[0]), k=1000))
        indices_spe = (random.sample(range(0,general_data_spe.shape[0]), k=1000))
        data = {}
        data["general-gen"] = np.take(general_data_gen, indices_gen, axis=0) # books: 430923 clothing: 117499
        data["control-gen"] = np.take(control_data_gen, indices_gen, axis=0)
        data["general-spe"] = np.take(general_data_spe, indices_spe, axis=0) # books: 430923 clothing: 117499
        data["control-spe"] = np.take(control_data_spe, indices_spe, axis=0)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general-gen', 'control-gen', 'general-spe', 'control-spe'], ['3', (5,2), '+', '1'], ["blue", 'red', 'cyan', 'magenta']):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==4:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1], l[2], l[3]]
            h = [h[0], h[1], h[2], h[3]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=12.5,
                            framealpha=0.6,
                            markerscale=1)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.axis('off')
        fig.savefig("trial_final_layer_representation_mask"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=600,
                    transparent=True)

        plt.clf()

def plot_final_layer_representation_with_mask1():
    # random.seed(30)
    # indices = (random.sample(range(0,117499),k=2500)) # books: 430923 clothing: 117499

    plt.figure(dpi=600)
    exp_names1 = ['10_model_10_data','10_model_100_data','100_model_10_data','100_model_100_data']
    exp_names2 = ['new_control_10_model_10_data','new_control_10_model_100_data','new_control_100_model_10_data','new_control_100_model_100_data']
    epoch1s = ['501','501','251','131']
    epoch2s = ['501','501','101','181'] # books
    # epoch2s = ['501','501','501','501','501'] # clothing
    for i in range(4):
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
        size = X_3d.shape[0]//2

        general_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.general'
        specific_mask_file = f'/disk/ocean/zheng/summarization_svcca/data/AmazonReviews/{MODEL_CAT2}/Test_2500_{MODEL_CAT2}.txt.specific'
        with open(general_mask_file) as f:
            word_mask_list_gen = []
            for line in f.readlines():
                word_mask_list_gen += [int(x) for x in line.strip().split()]
        word_mask_gen = np.array(word_mask_list_gen, dtype=bool)
        assert len(word_mask_gen) == acts1.shape[0] # sanity check

        with open(specific_mask_file) as f:
            word_mask_list_spe = []
            for line in f.readlines():
                word_mask_list_spe += [int(x) for x in line.strip().split()]
        word_mask_spe = np.array(word_mask_list_spe, dtype=bool)
        assert len(word_mask_spe) == acts1.shape[0] # sanity check

        general_data = X_3d[:size]
        general_data_gen = general_data[word_mask_gen]
        general_data_spe = general_data[word_mask_spe]

        control_data = X_3d[size:]
        control_data_gen = control_data[word_mask_gen]
        control_data_spe = control_data[word_mask_spe]
        
        random.seed(30)
        indices_gen = (random.sample(range(0,general_data_gen.shape[0]), k=1000))
        indices_spe = (random.sample(range(0,general_data_spe.shape[0]), k=1000))
        data = {}
        data["general-general"] = np.take(general_data_gen, indices_gen, axis=0) # books: 430923 clothing: 117499
        data["control-general"] = np.take(control_data_gen, indices_gen, axis=0)
        data["general-specific"] = np.take(general_data_spe, indices_spe, axis=0) # books: 430923 clothing: 117499
        data["control-specific"] = np.take(control_data_spe, indices_spe, axis=0)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, marker, color in zip(['general-general', 'control-general', 'general-specific', 'control-specific'], ['3', (5,2), '+', '1'], ["blue", 'red', 'cyan', 'magenta']):
            X_temp = data[label]
            ax.scatter(x=X_temp[:, 0], y=X_temp[:, 1],
                    label=label,
                    marker=marker,
                    color=color,
                    alpha=0.5)
        if i==3:
            legend = ax.legend()
            h, l = ax.get_legend_handles_labels()
            l = [l[0], l[1], l[2], l[3]]
            h = [h[0], h[1], h[2], h[3]]
            legend = ax.legend(h,
                            l,
                            loc='upper right',
                            fontsize=12.5,
                            framealpha=0.6,
                            markerscale=1)
            for lh in legend.legendHandles:
                lh.set_alpha(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.axis('off')
        fig.savefig("trial_final_layer_representation_mask-"+str(i)+".pdf",
                    format='pdf',
                    bbox_inches='tight',
                    dpi=600,
                    transparent=True)

        plt.clf()

if __name__ == '__main__':
    plot_embedding_weights2()
    # plot_embedding_layer_representation()
    # plot_embedding_layer_representation_with_mask1()
    # plot_embedding_layer_representation_with_mask_data()
    # plot_five_embedding_weights()
    # plot_final_layer_weights()
    # plot_final_layer_representation()
    # plot_final_layer_representation_with_mask1()