import csv
from json.tool import main
from collections import defaultdict
import ssl
from tkinter import font
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
matplotlib.rcParams['figure.dpi'] = 600

def read_out_file_backup(file, csv_file, end_epoch=202, layers=13):
    with open(file, 'r') as f:
        lines = f.readlines()
        results = lines[4::5] # this select every 5th line from the file which contains the actual svcca score
    processed_results = [float(res.strip().split()[1]) for res in results]
    # print(len(processed_results))
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        header_row = ['', 'epoch_0'] + ['epoch_'+str(i) for i in range(1,end_epoch,10)]
        # print(len(header_row)-1)
        # print(header_row)
        writer.writerow(header_row)
        for i in range(layers):
            # first go through each layer
            start_idx = (i*(len(header_row)-1)) # each layer, there are 22 (for epoch end 202) checkpoints saved, general formula is len(header_row)-1
            end_idx = ((i+1)*(len(header_row)-1))
            layer_result = processed_results[start_idx:end_idx]
            # print(layer_result)
            layer_row = ['layer_'+str(i)] + layer_result
            writer.writerow(layer_row)

def read_out_file_old(file, csv_file, end_epoch=202, layers=[0,1,2,3,4,5,6,7,8,9,10,11,12]):
    with open(file, 'r') as f:
        lines = f.readlines()
        results = lines[5::6] # this select every 5th line from the file which contains the actual svcca score
    processed_results = [float(res.strip().split()[1]) for res in results]
    # print(len(processed_results))
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        header_row = ['', 'epoch_0'] + ['epoch_'+str(i) for i in range(1,end_epoch,10)]
        # print(len(header_row)-1)
        # print(header_row)
        writer.writerow(header_row)
        for i, l in enumerate(layers):
            # first go through each layer
            start_idx = (i*(len(header_row)-1)) # each layer, there are 22 (for epoch end 202) checkpoints saved, general formula is len(header_row)-1
            end_idx = ((i+1)*(len(header_row)-1))
            layer_result = processed_results[start_idx:end_idx]
            # print(layer_result)
            layer_row = ['layer_'+str(l)] + layer_result
            writer.writerow(layer_row)

def read_out_file_dynamics(file, file_out, model_sizes=5, data_sizes=4, end_epoch=[202], layers=[0,1,2,3,4,5,6,7,8,9,10,11,12]):
    ms = [10,25,50,75,100]
    ds = [10,50,100,200]
    with open(file, 'r') as f:
        lines = f.readlines()
        results = lines[5::6] # this select every 5th line from the file which contains the actual svcca score
    processed_results = [float(res.strip().split()[1]) for res in results]
    with open(file_out, 'wb') as f:
        temps = []
        for m in range(model_sizes):
            for d in range(data_sizes):
                for _, l in enumerate(layers):
                    temp = []
                    result = processed_results.pop(0) # this is for epoch 0
                    temp.append(result)
                    # following is for epoch 1, 11, 21, ...
                    # print(m*data_sizes+d)
                    for _ in range(1, end_epoch[m*data_sizes+d]+1, 10):
                        result = processed_results.pop(0)
                        temp.append(result)
                    # f.write('m='+str(ms[m])+' ')
                    # f.write('d='+str(ds[d])+' ')
                    # f.write('layer='+str(l)+'\t')
                    # f.write(','.join(str(round(item,3)) for item in temp))
                    # f.write('\n')
                    temps.append(temp)
        pickle.dump(temps, f)
    assert processed_results == [] # sanity check that all results have been poped out


def read_out_file_domain(file, csv_file, domains, layers, model_sizes=5, data_sizes=4, title='plz put a title here', file_name='plot'):
    # this func can process out file for general model domains compare with oracle/control models
    with open(file, 'r') as f:
        lines = f.readlines()
        results = lines[5::6] # this select every 6th line from the file which contains the actual svcca score
    processed_results = [float(res.strip().split()[1]) for res in results]
    # print(len(processed_results))
    results_dict = defaultdict(dict) # {'layer0': {'domain1':[1,1,1,2,2,2,3,3,3]}} data x model
    for m in range(model_sizes):
        for d in range(data_sizes):
            for domain in (domains):
                for l in layers:
                    result = processed_results.pop(0)
                    layer_name = 'layer'+str(l)
                    if domain in results_dict[layer_name].keys():
                        results_dict[layer_name][domain].append(result)
                    else:
                        results_dict[layer_name][domain] = [result]
    # print((domains))
    assert processed_results == [] # sanity check that all results have been poped out
    plot_svcca_heatmap_individual(results_dict, model_sizes, data_sizes, domains, layers, title, file_name) 

def plot_svcca_heatmap(attention_dict, model_sizes, data_sizes, domains, layers, title, file_name):
    # x_ticks = ['model'+str(i+1) for i in range(model_sizes)]
    # y_ticks = ['data'+str(i+1) for i in range(data_sizes)]
    y_ticks = ['10% model', '25% model', '50% model', '75% model', '100% model']
    x_ticks = ['10% data', '50% data', '100% data', '200% data']

    for i in layers:
        layer_name = 'layer'+str(i)
        if len(domains) <= 3:
            fig, axs = plt.subplots(nrows=1, ncols=len(domains), figsize=(12,8))
            for m, ax in enumerate(axs):
                domain = domains[m]
        
                data = np.array(attention_dict[layer_name][domain])
                new_data = [data[j:j+data_sizes] for j in range(0, len(data), data_sizes)]
                
                im = ax.imshow(new_data, cmap=plt.get_cmap('Pastel1'))
                # Create colorbar
                cbar = ax.figure.colorbar(im, ax=ax, shrink=0.4)
                cbar.ax.set_ylabel('correlation', rotation=-90, va="bottom")

                # Show all ticks and label them with the respective list entries
                ax.set_xticks(np.arange(len(x_ticks)), labels=x_ticks)
                ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                # file_name = ''
                ax.set_title(domain)

                for (jj,ii),label in np.ndenumerate(new_data):
                    ax.text(ii,jj,round(label,3),ha='center',va='center',color='black')
        else:
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
            fig.delaxes(axs[1,2]) #The indexing is zero-based here
            for m, row in enumerate(axs):
                for n, ax in enumerate(row):
                    if m*len(axs) + n >= len(domains): continue # this will pass the 6th domain since we don't have one
                    else:
                        domain = domains[m*len(axs) + n]
                
                        data = np.array(attention_dict[layer_name][domain])
                        new_data = [data[j:j+data_sizes] for j in range(0, len(data), data_sizes)]
                        
                        im = ax.imshow(new_data, cmap=plt.get_cmap('Pastel2'))
                        # Create colorbar
                        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.4)
                        cbar.ax.set_ylabel('correlation', rotation=-90, va="bottom")

                        # Show all ticks and label them with the respective list entries
                        ax.set_xticks(np.arange(len(x_ticks)), labels=x_ticks)
                        ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)

                        # Rotate the tick labels and set their alignment.
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                        # file_name = ''
                        ax.set_title(domain)

                        for (jj,ii),label in np.ndenumerate(new_data):
                            ax.text(ii,jj,round(label,3),ha='center',va='center',color='black')
        # fig.suptitle(title + str(i))
        fig.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.show() 
        # plt.savefig(f'figures/{file_name}{i}.pdf')

def plot_svcca_heatmap_individual(attention_dict, model_sizes, data_sizes, domains, layers, title, file_name):
    # x_ticks = ['model'+str(i+1) for i in range(model_sizes)]
    # y_ticks = ['data'+str(i+1) for i in range(data_sizes)]
    y_ticks = ['10%', '25%', '50%', '75%', '100%']
    x_ticks = ['10%', '50%', '100%', '200%']

    for i in layers:
        layer_name = 'layer'+str(i)
        for _, domain in enumerate(domains):
            data = np.array(attention_dict[layer_name][domain])
            new_data = [data[j:j+data_sizes] for j in range(0, len(data), data_sizes)]
            
            # im = plt.imshow(new_data, cmap=plt.get_cmap('Greens'))
            # # Create colorbar
            # cbar = plt.colorbar(im, shrink=0.8)
            # cbar.ax.set_ylabel('SVCCA', rotation=-90, va="bottom")
            plt.figure(figsize=(5.5,5))
            sns.set(font_scale=1.3)
            ax = sns.heatmap(new_data, cmap=plt.get_cmap('RdYlGn'), annot=True, yticklabels=y_ticks, xticklabels=x_ticks, vmin=0., vmax=1)
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel('SVCCA similarity', rotation=-90, va="bottom")
            # Show all ticks and label them with the respective list entries
            # Rotate the x-tick labels and set their alignment.
            # plt.xticks(np.arange(len(x_ticks)), labels=x_ticks, rotation=0, ha="center", rotation_mode="anchor")
            # plt.yticks(np.arange(len(y_ticks)), labels=y_ticks)
            plt.yticks(rotation=0)

            plt.xlabel('Data size')
            plt.ylabel('Model size')

            # plt.set_title(domain)

            # for (jj,ii),label in np.ndenumerate(new_data):
            #     plt.text(ii,jj,round(label,3),ha='center',va='center',color='black')
            # plt.show()
            plt.tight_layout()
            # plt.savefig(f'figures/{file_name}_domain_{domain}_layer{i}.png')
            plt.savefig(f'figures/{file_name}_domain_{domain}_layer{i}.pdf')
            plt.clf()
            # exit()

def plot_svcca_heatmap1(attention_dict, model_sizes, data_sizes, domains, layers, title):
    # x_ticks = ['model'+str(i+1) for i in range(model_sizes)]
    # y_ticks = ['data'+str(i+1) for i in range(data_sizes)]
    y_ticks = ['10% model', '50% model', '100% model']
    x_ticks = ['10% data', '50% data', '100% data', '200% data']

    for i in layers:
        layer_name = 'layer'+str(i)
        fig, ax = plt.subplots()   
        domain = domains[0]

        data = np.array(attention_dict[layer_name][domain])
        new_data = [data[j:j+data_sizes] for j in range(0, len(data), data_sizes)]
        
        im = ax.imshow(new_data)
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.4)
        cbar.ax.set_ylabel('correlation', rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(x_ticks)), labels=x_ticks)
        ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # file_name = ''
        ax.set_title(domain)

        for (jj,ii),label in np.ndenumerate(new_data):
            ax.text(ii,jj,round(label,3),ha='center',va='center',color='r')
        fig.suptitle(title + ' for layer ' + str(i))
        fig.tight_layout()
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.show()  

def read_out_file_new_domain(file, csv_file, domains, layers, model_sizes=3, data_sizes=4, title='suptitle'):
    # this func can process out file for FT on new domain using existing general model or train from scratch mixing new domain with existing domain
    with open(file, 'r') as f:
        lines = f.readlines()
        results = lines[5::6] # this select every 6th line from the file which contains the actual svcca score
    processed_results = [float(res.strip().split()[1]) for res in results]
    # print(len(processed_results))
    results_dict = defaultdict(dict) # {'layer0': {'domain1':[1,1,1,2,2,2,3,3,3]}} data x model
    for m in range(model_sizes):
        for d in range(data_sizes):
            for domain in (domains):
                for l in layers:
                    result = processed_results.pop(0)
                    layer_name = 'layer'+str(l)
                    if domain in results_dict[layer_name].keys():
                        results_dict[layer_name][domain].append(result)
                    else:
                        results_dict[layer_name][domain] = [result]
    # print((domains))
    assert processed_results == [] # sanity check that all results have been poped out
    plot_svcca_heatmap1(results_dict, model_sizes, data_sizes, domains, layers, title)

def plot_attention_heatmap(attention_dict):
    x_ticks = ['head'+str(i+1) for i in range(12)]
    y_ticks = ['layer'+str(i+1) for i in range(12)]
    for i in range(22):
        if i > 0: epoch_num = (i-1)*10 + 1 # 1, 11, 21, 31, 41, ..., 201
        else: epoch_num = 0
        epoch_name = 'epoch'+str(epoch_num)
        data = np.array(attention_dict[epoch_name])
        fig, ax = plt.subplots()
        im = ax.imshow(data)
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('correlation', rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(x_ticks)), labels=x_ticks)
        ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        title = 'Attention correlation between models for epoch '+str(epoch_num)
        file_name = ''
        ax.set_title(title)
        fig.tight_layout()
        plt.show()

def read_attention_out_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        results = lines[1::2] # this select every 2nd line from the file which contains the actual correlation
    processed_results = [float(res.strip().split()[-1]) for res in results]
    # print(len(processed_results))
    results_dict = defaultdict(list) # {'epoch0':[[1,1,1],[1,1,1]]} 
    # results_dict = {} # {'epoch0':[[1,1,1],[1,1,1]]} 
    for i in range(12):
        # first go through each layer
        start_idx = (i*264) # each layer, there are 22 checkpoints * 12 heads = 264 correlations saved
        end_idx = ((i+1)*264)
        layer_result = processed_results[start_idx:end_idx]
        for j in range(22):
            # then go through epochs
            start = j*12
            end = (j+1)*12
            epoch_result = layer_result[start:end]
            if j > 0: epoch_num = (j-1)*10 + 1 # 1, 11, 21, 31, 41, ..., 201
            else: epoch_num = 0
            epoch_name = 'epoch'+str(epoch_num)
            results_dict[epoch_name].append(epoch_result)
    # print(results_dict)
    plot_attention_heatmap(results_dict)

def process_out_file_top50(file, removed_ones):
    correct_lines = []
    with open(file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if 'SVD done' in line:
                if int(line.split()[0].strip('top')) not in removed_ones:
                    correct_lines += lines[i-4:i+2]
    return correct_lines

def get_tf_counts(file, removed_ones):
    correct_lines = []
    with open(file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i+1 not in removed_ones:
                correct_lines.append(int(line.strip()))
    return correct_lines

if __name__ == '__main__':
    # # this is for training/learning dynamics with just one exp setting
    # read_out_file_old('out/train_dynamic_book_oracle.txt',
    #             'out/train_dynamic_book_oracle.csv',
    #             end_epoch=132,
    #             layers=[0,12])

    # read_out_file_dynamics('out_new/books_dynamic_all_model_all_data_with_general_mask_top5_control_layers_0_12_new.txt',
    # 'out_new/books_dynamic_all_model_all_data_with_general_mask_top5_control_layers_0_12_new.txt.pkl',
    # end_epoch=[501,501,501,501, 501,501,501,501, 501,501,501,501, 501,201,201,201, 251,151,131,131],
    # layers=[0,12])


    # read_out_file_domain('out_new/all_model_all_data_with_general_mask_top5_control_layers_0_12_new.txt',
    #                      '',
    #                      domains=['Books','Clothing_Shoes_and_Jewelry','Electronics','Home_and_Kitchen','Movies_and_TV'],
    #                      layers=[0,12],
    #                      title='SVCCA correlation between gereral and control model with freq mask for layer ',
    #                      file_name='general_vs_control_with_general_words')

    # read_out_file_domain('out_new/books_movies_all_model_all_data_with_freq001_mask_top5_control_layers_0_12.txt',
    #                      '',
    #                      domains=['Books', 'Movies_and_TV'],
    #                      layers=[0,12],
    #                      title='SVCCA correlation between gereral and control model with freq mask for layer ',
    #                      file_name='general_vs_control_with_freq001_words')

    # read_out_file_domain('out_new/books_all_model_all_data_with_freq001_mask_top5_control_layers_0_12.txt',
    #                      '',
    #                      domains=['Books'],
    #                      layers=[0,12],
    #                      title='SVCCA correlation between gereral and control model with freq mask for layer ',
    #                      file_name='general_vs_control_with_freq001_words')

    # read_out_file_new_domain('out/all_model_all_data_top5sports_oracle.txt',
    #                      '',
    #                      domains=['Sports_and_Outdoors'],
    #                      layers=[0,12],
    #                      title='SVCCA for new domain between model using top5 ckpt then FT and oracle')

    # read_out_file_new_domain('out/all_model_all_data_top6sports_oracle.txt',
    #                      '',
    #                      domains=['Sports_and_Outdoors'],
    #                      layers=[0,12],
    #                      title='SVCCA for new domain between general and oracle model')
    
    # read_attention_out_file('out/corr_top5_seed1_book_seed1_all_attentions.txt')


    with open('out_new/books_dynamic_all_model_all_data_top5_control_layers_0_12.txt.pkl','rb') as f:
        data = pickle.load(f)
    # print(data)
    with open('out_new/books_dynamic_all_model_all_data_with_general_mask_top5_control_layers_0_12_new.txt.pkl','rb') as f:
        data_general = pickle.load(f)
    with open('out_new/books_dynamic_all_model_all_data_with_specific_mask_top5_control_layers_0_12_new.txt.pkl','rb') as f:
        data_specific = pickle.load(f)

    # # # a = [0.83,0.833,0.863,0.905,0.935,0.954,0.958,0.955,0.952,0.948,0.942,0.933,0.924,0.916,0.909,0.902,0.896,0.889,0.883,0.877,0.872,0.867,0.862,0.857,0.853,0.848,0.844,0.84,0.837,0.833,0.83,0.827,0.824,0.821,0.819,0.816,0.815,0.813,0.811,0.81,0.808,0.807,0.806,0.805,0.804,0.804,0.803,0.803,0.802,0.802,0.802,0.802]
    # # # b = [0.834,0.836,0.864,0.9,0.93,0.952,0.958,0.958,0.953,0.946,0.94,0.935,0.928,0.922,0.915,0.906,0.898,0.889,0.882,0.874,0.871,0.864,0.859,0.854,0.849,0.845,0.841,0.837,0.834,0.831,0.828,0.824,0.821,0.819,0.817,0.814,0.812,0.81,0.809,0.807,0.806,0.804,0.803,0.802,0.801,0.8,0.799,0.799,0.799,0.798,0.798,0.798]
    ms = [10,25,50,75,100]
    ds = [10,50,100,200]

    # # portrait full 
    # fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(10,10))
    # for i, row in enumerate(ax):
    #     for j, col in enumerate(row):
    #         # j += 2
    #         idx = i * len(ds) + j
    #         idx0 = idx * 2 # for layer 0
    #         idx12 = idx * 2 + 1 # for layer 12
    #         col.plot(data[idx0],color='r',ls='-', label='layer0')
    #         col.plot(data[idx12],color='r',ls='--', label='layer12')
    #         col.plot(data_general[idx0],color='b',ls='-', label='layer0')
    #         col.plot(data_general[idx12],color='b',ls='--', label='layer12')
    #         col.plot(data_specific[idx0],color='g',ls='-', label='layer0')
    #         col.plot(data_specific[idx12],color='g',ls='--', label='layer12')
    #         if len(data[idx0]) < 20:
    #             col.set_xticks(np.arange(0,len(data[idx0]), step=5), [z*10 for z in range(0,len(data[idx0]),5)])
    #         else:
    #             col.set_xticks(np.arange(0,len(data[idx0]), step=10), [z*10 for z in range(0,len(data[idx0]),10)])
    #         # if i == len(ax)-1 and j == 3:
    #         #     col.legend(['embedding-all', 'embedding-general', 'embedding-specific', 'layer12-all', 'layer12-general', 'layer12-specific'], loc='upper left', bbox_to_anchor=(-1., -0.1))
    #         title = 'm='+str(ms[i])+'%'+' d='+str(ds[j])+'%'
    #         col.set_title(title)
    # fig.legend(['embedding-all-lexicon', 'layer12-all-lexicon', 'embedding-general-lexicon', 'layer12-general-lexicon', 'embedding-specific-lexicon', 'layer12-specific-lexicon'], loc='lower left', bbox_to_anchor=(0.15, 0.), ncol=3)
    # plt.tight_layout(rect=[0,0.06,1,1])
    # # plt.show() 
    # plt.savefig('figures/books_training_dynamics_full.pdf') 

    # portrait partial (without m=10 row and no all-lexicon) 
    ms = [25,50,75,100]
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10,8))
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            idx = i * len(ds) + j
            idx += 4 # inflating since we skip m=10 row (4 plots)
            idx0 = idx * 2 # for layer 0
            idx12 = idx * 2 + 1 # for layer 12
            # col.plot(data[idx0],color='r',ls='-', label='layer0')
            # col.plot(data[idx12],color='r',ls='--', label='layer12')
            col.plot(data_general[idx0],color='b',ls='-', label='layer0')
            col.plot(data_general[idx12],color='b',ls='--', label='layer12')
            col.plot(data_specific[idx0],color='r',ls='-', label='layer0')
            col.plot(data_specific[idx12],color='r',ls='--', label='layer12')
            if len(data[idx0]) < 20:
                col.set_xticks(np.arange(0,len(data[idx0]), step=5), [z*10 for z in range(0,len(data[idx0]),5)])
            else:
                col.set_xticks(np.arange(0,len(data[idx0]), step=10), [z*10 for z in range(0,len(data[idx0]),10)])
            # if i == len(ax)-1 and j == 3:
            #     col.legend(['embedding-all', 'embedding-general', 'embedding-specific', 'layer12-all', 'layer12-general', 'layer12-specific'], loc='upper left', bbox_to_anchor=(-1., -0.1))
            title = 'm='+str(ms[i])+'%'+' d='+str(ds[j])+'%'
            col.set_title(title)
    fig.legend(['embedding-general-lexicon', 'layer12-general-lexicon', 'embedding-specific-lexicon', 'layer12-specific-lexicon'], loc='lower left', bbox_to_anchor=(0.02, 0.), ncol=4)
    plt.tight_layout(rect=[0,0.06,1,1])
    # plt.show() 
    plt.savefig('figures/books_training_dynamics_partial.pdf') 

    # # portrait partial (only d=100 and d=200)
    # fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(10,12))
    # for i, row in enumerate(ax):
    #     for j, col in enumerate(row):
    #         j += 2
    #         idx = i * len(ds) + j
    #         idx0 = idx * 2 # for layer 0
    #         idx12 = idx * 2 + 1 # for layer 12
    #         col.plot(data[idx0],color='r',ls='-', label='layer0')
    #         col.plot(data_general[idx0],color='b',ls='-', label='layer0')
    #         col.plot(data_specific[idx0],color='g',ls='-', label='layer0')
    #         col.plot(data[idx12],color='r',ls='--', label='layer 12')
    #         col.plot(data_general[idx12],color='b',ls='--', label='layer12')
    #         col.plot(data_specific[idx12],color='g',ls='--', label='layer12')
    #         if i == len(ax)-1 and j == len(ds)-1:
    #             col.legend(['l0-all', 'l0-general', 'l0-specific', 'l12-all', 'l12-general', 'l12-specific'], loc='right', bbox_to_anchor=(1.5, 0.5))
    #         title = 'm='+str(ms[i])+' d='+str(ds[j])
    #         col.set_title(title)

            

    # # landscpae partial
    # plt.figure(figsize=(10,7))
    # for i in (range(5)):
    #     for j in (range(2)):
    #         zz = i+1+j*5
    #         j += 2
    #         idx = i * len(ds) + j
    #         idx0 = idx * 2 # for layer 0
    #         idx12 = idx * 2 + 1 # for layer 12
    #         ax = plt.subplot(2, 5, zz)
    #         plt.plot(data[idx0],color='r',ls='-', label='layer0')
    #         plt.plot(data_general[idx0],color='b',ls='-', label='layer0')
    #         plt.plot(data_specific[idx0],color='g',ls='-', label='layer0')
    #         plt.plot(data[idx12],color='r',ls='--', label='layer 12')
    #         plt.plot(data_general[idx12],color='b',ls='--', label='layer12')
    #         plt.plot(data_specific[idx12],color='g',ls='--', label='layer12')
    #         if i == len(ms)-3 and j == len(ds)-1:
    #             plt.legend(['layer0-all', 'layer0-general', 'layer0-specific', 'layer12-all', 'layer12-general', 'layer12-specific'], loc='lower center', bbox_to_anchor=(0.5, -0.8))
    #         title = 'm='+str(ms[i])+' d='+str(ds[j])
    #         ax.set_title(title)
        
    # plt.tight_layout()
    # # plt.show() 
    # plt.savefig('figures/books_training_dynamics_full.pdf') 



    # with open('out_new/zz0.txt', 'r') as f:
    #     lines = f.readlines()
    #     results = lines[5::6] # this select every 5th line from the file which contains the actual svcca score
    # processed_results = [float(res.strip().split()[1]) for res in results]
    # # print(len(processed_results))
    # # print(processed_results)
    # with open('out_new/zz0_counts.txt', 'r') as f:
    #     lines = f.readlines()
    # processed_counts = [float(res.strip()) for res in lines]
    # # print(len(processed_counts))
    # # print(processed_counts)
    # print(np.corrcoef(processed_results,processed_counts))
    # # plt.scatter(processed_counts,processed_results)
    # # # plt.scatter(processed_counts,processed_results)
    # # plt.show()


    # # top 50 words scatter plots for svcca and freq
    # plt.rc('font', size=18)
    # book_list = [2,16,17,47,48,49] # these top words does not have enough counts to calc svcca so will be removed
    # processed_results_all = process_out_file_top50('out_new/books_top50_svcca_per_word.txt', book_list)
    # processed_counts = get_tf_counts('top50_tf_counts.txt', book_list)
    # for i,m in enumerate([10,25,50,75,100]):
    #     for j,d in enumerate([10,50,100,200]):
    #         for k,l in enumerate([0,12]):
    #             idx = (i*(8) + j*2+k+1)
    #             block_line_length = (50-len(book_list))*6# (50-removed top_n words) * 6 lines are one block of results
    #             processed_results_cleaned = processed_results_all[(idx-1)*block_line_length : idx*block_line_length]
    #             # print(len(processed_results_cleaned))
    #             results = processed_results_cleaned[5::6] # this select every 5th line from the file which contains the actual svcca score
    #             processed_results = [float(res.strip().split()[1]) for res in results]
    #             # print(len(processed_results))
    #             assert len(processed_counts) == len(processed_results)
    #             # print(f'model_size={m},data_size={d},layer={l}')
    #             # print(np.corrcoef(processed_results,processed_counts)[0][1])
    #             plt.scatter(processed_counts[0:],processed_results[0:])
                
                
    #         # plt.title(f'model_size={m},data_size={d}')
    #         plt.legend(['embedding','layer12'], loc='best')
    #         plt.xlabel('Frequency')
    #         plt.ylabel('SVCCA similarity')
    #         plt.tight_layout()
    #         # plt.show()
    #         plt.savefig(f'figures/top50_svcca_freq_corr/m_{m}_d_{d}.pdf')
    #         plt.clf()
    

    # # plot the rainbow (all layers training dynamics)
    # books_data = [
    #     [0.999999987,0.999266043,0.988718023,0.969456993,0.95728586,0.945718163,0.934388688,0.923573201,0.912969893,0.902925341,0.893179655,0.884035005,0.874966239,0.866435845,0.858149347,0.8504832,0.843173122,0.836333158,0.829918791,0.82382282,0.817984574,0.812495865],
    #     [0.999999985,0.998915845,0.983747814,0.951424386,0.934388383,0.919480132,0.905367088,0.89219164,0.878733476,0.865717125,0.853485182,0.841936974,0.830675151,0.819592439,0.80951721,0.799721424,0.790688422,0.781716312,0.773931448,0.766254551,0.758888848,0.752432655],
    #     [0.999999983,0.997396678,0.976166701,0.920240065,0.897634951,0.878970718,0.861028438,0.843713922,0.827926985,0.812200956,0.797554265,0.783246337,0.77002296,0.757414668,0.745719966,0.734337955,0.724433278,0.713735262,0.704560776,0.696677594,0.688812559,0.681564897],
    #     [0.99999998,0.997011656,0.958856464,0.864229852,0.832611149,0.806395606,0.783340423,0.763052939,0.745581509,0.728398475,0.712818619,0.698283464,0.684804298,0.673276449,0.662103202,0.651331067,0.642425128,0.631985615,0.623542355,0.616819227,0.609987307,0.603051175],
    #     [0.999999977,0.995130954,0.940961323,0.810356765,0.773842034,0.744760789,0.719150569,0.69655762,0.678099303,0.660037596,0.644322488,0.629851312,0.616823425,0.606286433,0.5960246,0.585872832,0.578242195,0.568563651,0.560949673,0.555602003,0.549476551,0.543385327],
    #     [0.999999974,0.991397222,0.916748095,0.752677688,0.71470998,0.685711213,0.659178621,0.636436173,0.618333205,0.601141271,0.586550145,0.572982609,0.561499255,0.552426133,0.543199907,0.534524528,0.528509412,0.519767169,0.513432502,0.508970136,0.504073088,0.498656492],
    #     [0.99999997,0.988559187,0.885400311,0.690458425,0.652421193,0.626021226,0.601575212,0.581362174,0.566355566,0.551726494,0.540255816,0.529345276,0.520230301,0.512993712,0.50577091,0.499006974,0.494190761,0.487732933,0.48262895,0.479179385,0.475436966,0.471121976],
    #     [0.999999967,0.982979062,0.847590076,0.623676837,0.58751153,0.563690729,0.542645901,0.526643287,0.515339828,0.504536081,0.496407061,0.488761393,0.482589489,0.477380062,0.472592724,0.467811083,0.464255029,0.459897609,0.456351871,0.453673048,0.450770579,0.447583122],
    #     [0.999999963,0.977340644,0.802430631,0.576212919,0.5454448,0.526563239,0.509963546,0.498921247,0.490747277,0.483469015,0.477739875,0.472244678,0.468074134,0.46415281,0.460467825,0.457147589,0.454113228,0.45099565,0.448549335,0.445730682,0.443463193,0.44092431],
    #     [0.999999959,0.969202782,0.766952158,0.537163617,0.5086086,0.493376481,0.481299976,0.473857502,0.468010379,0.462716601,0.458609769,0.454465713,0.451199127,0.448203828,0.445343206,0.442636683,0.439899022,0.437536897,0.435227446,0.432617673,0.430752872,0.428396956],
    #     [0.999999957,0.960764096,0.732348183,0.508841992,0.482123317,0.46911322,0.459107807,0.453385012,0.448677858,0.444849988,0.441482809,0.437942625,0.435248292,0.432369104,0.42975013,0.427309698,0.424489796,0.422510018,0.420273718,0.417644935,0.415713516,0.413408492],
    #     [0.999999953,0.952292149,0.708905774,0.495984477,0.470269321,0.456897034,0.446996381,0.441387798,0.435788175,0.432147265,0.427919229,0.4243793,0.421491542,0.418420382,0.415671862,0.412884492,0.409875934,0.407874603,0.40521491,0.402374999,0.400256347,0.397779646],
    #     [0.999999949,0.942630894,0.693368088,0.489792386,0.464612014,0.450889782,0.44040443,0.434104202,0.427778996,0.423710527,0.418763426,0.41479462,0.41114149,0.407575634,0.404328934,0.401138694,0.397747326,0.395513452,0.392509418,0.38931283,0.386961813,0.384278996]
    # ]
    # clothing_data = [
    #     [0.999999981,0.998927987,0.984261166,0.967338346,0.95474611,0.943611968,0.933094833,0.922855166,0.913627965,0.904407178,0.895762257,0.887420541,0.879570255,0.87196999,0.864839984,0.858302206,0.851988883,0.845909636,0.840161691,0.834851164,0.829943621,0.825275525],
    #     [0.999999977,0.998110808,0.968369425,0.939831387,0.920909145,0.904326668,0.889313278,0.875073967,0.861482306,0.848254661,0.835770238,0.823599498,0.812203935,0.80126794,0.791559736,0.782038831,0.773652985,0.76530454,0.758248694,0.751220565,0.744792802,0.739221273],
    #     [0.999999975,0.99691487,0.928591061,0.878693023,0.85258151,0.831860954,0.812603742,0.794952615,0.778750148,0.762875168,0.748493439,0.734384339,0.72185128,0.710158994,0.699452588,0.688277481,0.68000409,0.670518689,0.663211677,0.656514837,0.649933899,0.644419483],
    #     [0.999999972,0.994017798,0.878785616,0.808764527,0.776532021,0.752145478,0.730619327,0.713355545,0.698147995,0.682539553,0.669395867,0.656265989,0.645123597,0.635175057,0.62564411,0.61647489,0.609435792,0.600816684,0.594704806,0.58967281,0.584275957,0.579381116],
    #     [0.999999969,0.989905175,0.832048269,0.743423956,0.70849289,0.682801191,0.660628542,0.643522724,0.628687919,0.613997845,0.60222773,0.590484922,0.581060426,0.572736849,0.564618178,0.55687299,0.551227146,0.543689733,0.538840114,0.534913652,0.530554673,0.526811535],
    #     [0.999999966,0.984009932,0.780735814,0.681128859,0.64664127,0.623124452,0.602108432,0.586352791,0.572997229,0.560269647,0.550600058,0.540289689,0.53243847,0.525770645,0.5186276,0.512643042,0.508264554,0.501582937,0.497860689,0.494445958,0.490977456,0.487820928],
    #     [0.999999962,0.977410732,0.726212436,0.624771055,0.594219143,0.574032634,0.555776583,0.542483855,0.531295087,0.520991401,0.513349621,0.505149992,0.498514636,0.49338634,0.487335126,0.482493279,0.478840571,0.473481777,0.470617768,0.4677293,0.464776117,0.46221821],
    #     [0.999999959,0.969069131,0.662334151,0.564632292,0.539404539,0.524417961,0.510705661,0.50171293,0.493753738,0.486714314,0.481275122,0.475527847,0.470778181,0.467182057,0.462808366,0.459153005,0.456498705,0.452414573,0.450352613,0.447790777,0.445272961,0.443203884],
    #     [0.999999956,0.958565287,0.608252382,0.521258665,0.501562931,0.490210454,0.479956904,0.473851866,0.467999241,0.462576846,0.458652948,0.454248133,0.450581502,0.447586615,0.444156104,0.440811383,0.438513626,0.435274723,0.433348181,0.430982323,0.428588037,0.426849236],
    #     [0.999999952,0.945617315,0.566544636,0.489749775,0.472902646,0.463907298,0.455525616,0.45082602,0.445893822,0.441695739,0.438145343,0.43424738,0.431185369,0.428228526,0.425075961,0.422101407,0.419697799,0.416806279,0.414915011,0.412600165,0.410490356,0.4086959],
    #     [0.99999995,0.932262909,0.533425467,0.46685549,0.450848978,0.442793565,0.435717249,0.431382143,0.427012687,0.423048099,0.419623942,0.415740551,0.412850126,0.40979992,0.40669197,0.403480459,0.401033836,0.398051744,0.396195004,0.393911837,0.391616183,0.389728925],
    #     [0.999999947,0.916968167,0.510020091,0.451122451,0.436232362,0.428247489,0.421026695,0.416444893,0.411817453,0.40776041,0.403974535,0.399896254,0.396703481,0.393354937,0.390045149,0.386474242,0.383497042,0.380341999,0.37821414,0.375764723,0.373481591,0.371313022],
    #     [0.999999943,0.901710548,0.496470081,0.443762949,0.429554194,0.421087247,0.413476786,0.408230629,0.402775892,0.398381624,0.39344249,0.389100236,0.385353937,0.381756643,0.378092226,0.373823815,0.370538343,0.367046754,0.364121142,0.361614237,0.358819223,0.35647856]
    # ]
    # data = [
    #     [0.999999986,0.999228311,0.986518604,0.967869773,0.955217328,0.944224119,0.934006353,0.92396311,0.914365823,0.905242299,0.896435118,0.888062979,0.87990779,0.871911907,0.864544632,0.857709062,0.851002257,0.844749206,0.838965834,0.833318072,0.828080851,0.82310756],
    #     [0.999999983,0.998468988,0.978550368,0.945478432,0.927522334,0.912177175,0.898034161,0.884569209,0.871788344,0.85997632,0.848040309,0.836858441,0.826221014,0.815676869,0.806283724,0.797181112,0.788575231,0.780435739,0.772676902,0.765594508,0.758919926,0.753018396],
    #     [0.999999981,0.997885241,0.964047851,0.909048923,0.884857027,0.864814905,0.84620037,0.829562549,0.813976779,0.798998508,0.784293368,0.770625924,0.758114828,0.746050832,0.735068883,0.723946623,0.715224263,0.704839921,0.696435974,0.689035731,0.681452411,0.67544011],
    #     [0.999999978,0.995937599,0.943406956,0.852856537,0.819721765,0.793497822,0.771311182,0.752653888,0.736490947,0.72039384,0.705568433,0.692668022,0.680486024,0.669266285,0.65885219,0.648680155,0.641360506,0.631263593,0.624149876,0.617796878,0.611177264,0.605954663],
    #     [0.999999975,0.994439533,0.917487633,0.797767066,0.760627737,0.731796695,0.707458145,0.686870017,0.669578135,0.653115192,0.638631881,0.626494435,0.615381174,0.60517853,0.596032752,0.58669509,0.580815725,0.571036542,0.565183629,0.559929027,0.554229263,0.549838571],
    #     [0.999999972,0.991449434,0.883141633,0.738739524,0.701197513,0.672861816,0.648915508,0.628849335,0.611806368,0.596749566,0.583816125,0.573034037,0.563032946,0.553994688,0.546469869,0.538408288,0.53412804,0.524993735,0.520466086,0.515721808,0.511149622,0.507399952],
    #     [0.999999968,0.987122328,0.840266995,0.6792702,0.644048054,0.618059411,0.595667088,0.577648292,0.562598812,0.550510339,0.539634793,0.531056766,0.523033179,0.515552212,0.509838662,0.50340448,0.500297268,0.492836152,0.489609491,0.485541982,0.481990852,0.478823189],
    #     [0.999999965,0.981261733,0.787502228,0.614512167,0.581748167,0.559358746,0.541161871,0.52807672,0.517424811,0.509039966,0.50158888,0.495979865,0.490556838,0.485465853,0.48144502,0.476997379,0.474708246,0.469494139,0.467265788,0.464071581,0.461440361,0.459014362],
    #     [0.999999961,0.97546904,0.737484396,0.563054928,0.534995204,0.517506577,0.504068202,0.494875866,0.487958281,0.482409164,0.477449364,0.473528144,0.469853167,0.466275073,0.463183397,0.459954922,0.458027876,0.454268196,0.452150957,0.449585373,0.447040755,0.444916878],
    #     [0.999999957,0.966300878,0.69866954,0.526437,0.499750749,0.485448073,0.475432426,0.468629081,0.463687125,0.459709091,0.455600985,0.452326051,0.449575778,0.446828062,0.443869644,0.441487807,0.439508259,0.43651238,0.434564683,0.43224898,0.429773076,0.42789756],
    #     [0.999999955,0.957671154,0.664421721,0.50014332,0.475754374,0.462980569,0.454969277,0.449866382,0.445586912,0.442457758,0.438824994,0.435792614,0.433390793,0.430699041,0.427916807,0.425487102,0.423291011,0.420517662,0.41835157,0.416051112,0.413538825,0.411322439],
    #     [0.999999951,0.947854319,0.636635511,0.483741572,0.460772995,0.448783699,0.441073235,0.43612545,0.431301144,0.428242056,0.424237255,0.421101779,0.41820529,0.415374674,0.412488714,0.409667167,0.407246636,0.404312441,0.401677127,0.399321601,0.396626726,0.394069679],
    #     [0.999999947,0.936788948,0.618074177,0.475168894,0.453356274,0.441103633,0.433190408,0.427702681,0.422622239,0.419145494,0.414221317,0.41046949,0.407124094,0.403843398,0.400636957,0.397138911,0.394326119,0.391017227,0.387822315,0.385348031,0.38206662,0.379279015]
    # ]
    # electronics_data = [
    #     [0.999999986,0.999237821,0.988079213,0.969337469,0.956616635,0.945433237,0.934175062,0.924008126,0.914043512,0.904502948,0.895512119,0.886652707,0.878157902,0.870370529,0.86303086,0.855733493,0.848768076,0.842395252,0.836228795,0.830492786,0.825105847,0.820035361],
    #     [0.999999984,0.99844084,0.981510658,0.948310502,0.930529989,0.915729707,0.901688291,0.88797369,0.875350102,0.862639205,0.850764529,0.839360287,0.828528563,0.818090307,0.808293839,0.799225142,0.790407021,0.782033487,0.774096715,0.766900116,0.759820078,0.753828765],
    #     [0.999999982,0.998007826,0.971558924,0.913892903,0.890851587,0.871165632,0.853352278,0.835912993,0.820298397,0.805006164,0.790389038,0.776568277,0.764062164,0.751492604,0.740346065,0.729576031,0.719919969,0.709835054,0.70100828,0.693092299,0.685672201,0.679108426],
    #     [0.999999979,0.996544473,0.955010928,0.857026687,0.824852981,0.798170021,0.776083728,0.756417627,0.739448975,0.722921518,0.708096499,0.69477173,0.681978787,0.669686106,0.659121446,0.649106626,0.640486215,0.630185454,0.622296456,0.615709354,0.609052629,0.603176793],
    #     [0.999999976,0.994446534,0.935107553,0.802596944,0.765900427,0.736749535,0.711487469,0.689935672,0.671923861,0.655039488,0.64022598,0.627378407,0.615634866,0.604013768,0.594629116,0.585351545,0.57812205,0.568286672,0.561532145,0.556052355,0.550145572,0.545359611],
    #     [0.999999973,0.991372,0.907688975,0.745001257,0.706585176,0.677653149,0.652469558,0.631213961,0.613518771,0.597296483,0.583770587,0.572533825,0.561884452,0.551171899,0.543293817,0.535054768,0.529251876,0.520279815,0.514848476,0.510200255,0.505333078,0.501317214],
    #     [0.999999969,0.987076559,0.874540372,0.684695512,0.64589172,0.619286324,0.595333641,0.576853008,0.561567575,0.547301083,0.536419007,0.527493239,0.51854435,0.509576219,0.504119813,0.497208446,0.492443516,0.485529544,0.481226347,0.477632617,0.473567321,0.470515523],
    #     [0.999999966,0.981827511,0.833361162,0.61875962,0.581683521,0.558830601,0.538624603,0.524513528,0.513175454,0.502611043,0.495494579,0.48954631,0.483730289,0.477367285,0.473802547,0.468690924,0.46571021,0.460658307,0.457779101,0.454951704,0.45208499,0.449765711],
    #     [0.999999962,0.974491603,0.788420795,0.565706838,0.533971749,0.515816559,0.500778174,0.490872548,0.483794725,0.476546825,0.471850382,0.467604491,0.463639555,0.459138104,0.45656089,0.452681805,0.450148091,0.446515323,0.444243877,0.441971064,0.439143508,0.437328027],
    #     [0.999999958,0.966999633,0.749972294,0.528314852,0.498995237,0.483960135,0.472288241,0.465107879,0.46011337,0.454795353,0.451687961,0.447992524,0.44522693,0.441679898,0.439596182,0.436237604,0.434121891,0.431154831,0.429160695,0.427114079,0.424522762,0.422845173],
    #     [0.999999956,0.958381981,0.717269165,0.500768695,0.475268916,0.463556893,0.454765504,0.449355528,0.445910687,0.441656819,0.439176139,0.435877873,0.433363471,0.430147559,0.428035931,0.425219331,0.422649383,0.419851713,0.417714544,0.415706189,0.412848862,0.411177106],
    #     [0.999999952,0.947750884,0.690050422,0.484448041,0.460202212,0.449558871,0.441162785,0.436239133,0.432740982,0.428768639,0.425868708,0.422289473,0.419386998,0.416381745,0.413930226,0.410994042,0.40819769,0.405204743,0.402642179,0.400486753,0.397522787,0.395872351],
    #     [0.999999948,0.937557883,0.672880431,0.47747967,0.4540817,0.442585863,0.434096915,0.428655885,0.424090709,0.419434077,0.415808778,0.411788122,0.408299758,0.40481376,0.401938503,0.398180889,0.395008581,0.391722108,0.388703578,0.386336793,0.3829508,0.381241651]
    # ]
    # plt.rc('font', size=12)
    # plt.figure(figsize=(7,4))
    # plt.plot(data[0], label='embedding')
    # plt.plot(data[1], label='layer1')
    # plt.plot(data[2], label='layer2')
    # plt.plot(data[3], label='layer3')
    # plt.plot(data[4], label='layer4')
    # plt.plot(data[5], label='layer5')
    # plt.plot(data[6], label='layer6')
    # plt.plot(data[7], label='layer7')
    # plt.plot(data[8], label='layer8')
    # plt.plot(data[9], label='layer9')
    # plt.plot(data[10], label='layer10', color='fuchsia')
    # plt.plot(data[11], label='layer11', color='lime')
    # plt.plot(data[12], label='layer12', color='blue')
    # plt.xlabel('epoch')
    # plt.xticks(np.arange(0,21,step=5), [0,50,100,150,200])
    # plt.ylabel('SVCCA similarity')
    # plt.legend(loc='upper left', bbox_to_anchor=(1., 1.))
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig('figures/home_training_dynamics_all_layers.pdf') 


