import csv
from json.tool import main
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

def read_out_file_domain(file, csv_file, domains, layers, model_sizes=3, data_sizes=4):
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
    # print(results_dict)clear
    plot_svcca_heatmap(results_dict, model_sizes, data_sizes, domains, layers)

    # with open(csv_file, 'w') as f:
    #     writer = csv.writer(f)
    #     header_row = ['', 'epoch_0'] + ['epoch_'+str(i) for i in range(1,end_epoch,10)]
    #     # print(len(header_row)-1)
    #     # print(header_row)
    #     writer.writerow(header_row)
    #     for i in range(layers):
    #         # first go through each layer
    #         start_idx = (i*(len(header_row)-1)) # each layer, there are 22 (for epoch end 202) checkpoints saved, general formula is len(header_row)-1
    #         end_idx = ((i+1)*(len(header_row)-1))
    #         layer_result = processed_results[start_idx:end_idx]
    #         # print(layer_result)
    #         layer_row = ['layer_'+str(i)] + layer_result
    #         writer.writerow(layer_row)

def plot_svcca_heatmap(attention_dict, model_sizes, data_sizes, domains, layers):
    # x_ticks = ['model'+str(i+1) for i in range(model_sizes)]
    # y_ticks = ['data'+str(i+1) for i in range(data_sizes)]
    y_ticks = ['10% model', '50% model', '100% model']
    x_ticks = ['10% data', '50% data', '100% data', '200% data']

    for i in layers:
        layer_name = 'layer'+str(i)
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
        fig.delaxes(axs[1,2]) #The indexing is zero-based here
        for m, row in enumerate(axs):
            for n, ax in enumerate(row):
                if m*model_sizes + n >= len(domains): continue # this will pass the 6th domain since we don't have one
                else:
                    domain = domains[m*model_sizes + n]
            
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
        fig.suptitle('SVCCA correlation between gereral and oracle model for layer ' + str(i))
        fig.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.show()  

def read_out_file_new_domain(file, csv_file, end_epoch=202, layers=13):
    # this func can process out file for FT on new domain using existing general model or train from scratch mixing new domain with existing domain
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


if __name__ == '__main__':
    # read_out_file('out/100_data_svcca_top5_seed1_home_seed1_all_layers.txt',
    #             'out/100_data_svcca_top5_seed1_home_seed1_all_layers.csv',
    #             end_epoch=202,
    #             layers=13)
    read_out_file_domain('out/all_model_all_data_top5_oracle.txt',
                         '',
                         domains=['Books','Clothing_Shoes_and_Jewelry','Electronics','Home_and_Kitchen','Movies_and_TV'],
                         layers=[0,12])
    # read_attention_out_file('out/corr_top5_seed1_book_seed1_all_attentions.txt')