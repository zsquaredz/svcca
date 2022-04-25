import csv
from json.tool import main
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def read_out_file(file, csv_file, end_epoch=202, layers=13):
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
    read_out_file('out/100_data_svcca_top5_seed1_home_seed1_all_layers.txt',
                'out/100_data_svcca_top5_seed1_home_seed1_all_layers.csv', 
                end_epoch=202,
                layers=13)
    # read_attention_out_file('out/corr_top5_seed1_book_seed1_all_attentions.txt')