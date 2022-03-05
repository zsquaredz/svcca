import csv
from json.tool import main

def read_out_file(file, csv_file):
    with open(file, 'r') as f:
        lines = f.readlines()
        results = lines[4::5] # this select every 5th line from the file which contains the actual svcca score
    processed_results = [float(res.strip().split()[1]) for res in results]
    # print(len(processed_results))
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        header_row = ['', 'epoch_0'] + ['epoch_'+str(i) for i in range(1,202,10)]
        # print(header_row)
        writer.writerow(header_row)
        for i in range(13):
            # first go through each layer
            start_idx = (i*22) # each layer, there are 22 checkpoints saved
            end_idx = ((i+1)*22)
            layer_result = processed_results[start_idx:end_idx]
            # print(layer_result)
            layer_row = ['layer_'+str(i)] + layer_result
            writer.writerow(layer_row)

if __name__ == '__main__':
    read_out_file('out/svcca_top5_seed1_home_seed1_all_layers.txt',
                    'out/svcca_top5_seed1_home_seed1_all_layers.csv')