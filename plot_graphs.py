from glob import glob
import os, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml 

stream = open('params.yml', 'r')
mydict = yaml.full_load(stream)

base_dir = mydict['folder']
base_input_dir = os.path.join(base_dir, 'test_outputs')

verbose = 0

def get_data_for_single_sample_count(sample_dir_path):
    all_csv_files = [file
                 for path, subdir, files in os.walk(sample_dir_path)
                 for file in glob(os.path.join(path, '*.csv'))]
    
    returns_list = []
    for csv_file in all_csv_files:
        results_df = pd.read_csv(csv_file)
        results_df = results_df.loc[:, ~results_df.columns.str.contains('^Unnamed')]
        
        return_list = results_df.groupby(['episode'])['reward'].sum()
        returns_list.append(return_list)
        return returns_list 
    
sample_trial_return_mean_list, sample_trial_return_std_list = [], []

sanitizing_sample_count_list = [2048, 4096, 12288, 16384, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 28672, 32768, 36864]
original_sample_count_list= [3.375283704702059,4.276720752889558,11.79744797045387,11.8966352184745,27.58572403773783,21.65782733250483,34.794303151237216,41.32899243259643,250.8999784121872,346.230592848016,444.5164038204594,478.2984135874861,446.0071298127695,398.21929717082963,434.7639635224309,412.9690244058976]
original_std_list= [5,5,5,5,23.5,0,10,0,50,72,20.4,35.3,30.8,25.4,41.3,57]
original_std_list= np.array(original_std_list)
print(sanitizing_sample_count_list)

for num_sample in sanitizing_sample_count_list:
    if(verbose):
        print('Clean samples : ', num_sample)
    sample_base_dir = os.path.join(base_input_dir, 'sanitized/clean_samples_'+str(num_sample)+'/poison_2000')     
    return_mean_list = []
    for trial in os.listdir(sample_base_dir):
        trial_csv_file = os.path.join(sample_base_dir, trial, 'log', 'csv_data.csv')

        results_csv = pd.read_csv(trial_csv_file)
        return_str = results_csv.loc[results_csv.shape[0]-1, 'return_list']
        return_list = [int(num) for num in re.findall('[0-9]+', return_str)]
        
        return_mean, return_std = np.mean(return_list), np.std(return_list)
        return_mean_list.append(return_mean)
        
        if(verbose):
            print('Mean : {0:2.4f}'.format(return_mean))
    sample_trial_return_mean_list.append(np.mean(return_mean_list)), sample_trial_return_std_list.append(np.std(return_mean_list))

st_return_mean_list, st_return_std_list = sample_trial_return_mean_list, sample_trial_return_std_list

sample_base_dir = os.path.join(base_input_dir, 'non_sanitized/no_poison') 

return_mean_list = []
for trial in os.listdir(sample_base_dir):
    trial_csv_file = os.path.join(sample_base_dir, trial, 'log', 'csv_data.csv')

    results_csv = pd.read_csv(trial_csv_file)
    return_str = results_csv.loc[results_csv.shape[0]-1, 'return_list']
    return_list = [int(num) for num in re.findall('[0-9]+', return_str)]

    return_mean, return_std = np.mean(return_list), np.std(return_list)
    return_mean_list.append(return_mean)

    if(verbose):
        print('Mean : {0:2.4f}'.format(return_mean))

tc_return_mean_list, tc_return_std_list = [np.mean(return_mean_list)]*len(sanitizing_sample_count_list), [np.std(return_mean_list)]*len(sanitizing_sample_count_list)
sample_base_dir = os.path.join(base_input_dir, 'non_sanitized/poison_2000') 

return_mean_list = []
for trial in os.listdir(sample_base_dir):
    print(trial)
    trial_csv_file = os.path.join(sample_base_dir, trial, 'log', 'csv_data.csv')

    results_csv = pd.read_csv(trial_csv_file)
    return_str = results_csv.loc[results_csv.shape[0]-1, 'return_list']
    return_list = [int(num) for num in re.findall('[0-9]+', return_str)]

    return_mean, return_std = np.mean(return_list), np.std(return_list)
    return_mean_list.append(return_mean)
    if(verbose):
        print('Mean : {0:2.4f}'.format(return_mean))

tt_return_mean_list, tt_return_std_list = [np.mean(return_mean_list)]*len(sanitizing_sample_count_list), [np.std(return_mean_list)]*len(sanitizing_sample_count_list)

fig, ax = plt.subplots(figsize=(12,8), dpi=80)
plt.rcParams['font.size'] = '25'

st_return_mean_list, st_return_std_list = np.array(st_return_mean_list), np.array(st_return_std_list)
tc_return_mean_list, tc_return_std_list = np.array(tc_return_mean_list), np.array(tc_return_std_list)
tt_return_mean_list, tt_return_std_list = np.array(tt_return_mean_list), np.array(tt_return_std_list)


plt.plot(sanitizing_sample_count_list, tc_return_mean_list, color='blue', label='backdoor in clean env')
plt.plot(sanitizing_sample_count_list, tt_return_mean_list, color='brown', label='backdoor in trigger env')
plt.plot(sanitizing_sample_count_list, st_return_mean_list,  marker='.', linestyle='-', markersize=12, color='orange',  label='sanitized in-distribution trigger env')
plt.plot(sanitizing_sample_count_list, original_sample_count_list, marker='.', linestyle='-', markersize=12, color='green',  label='sanitized in simple trigger env')

plt.fill_between(sanitizing_sample_count_list, tc_return_mean_list-tc_return_std_list, tc_return_mean_list+tc_return_std_list, facecolor='blue', alpha=0.25)
plt.fill_between(sanitizing_sample_count_list, tt_return_mean_list-tt_return_std_list, tt_return_mean_list+tt_return_std_list, facecolor='brown', alpha=0.25)
plt.fill_between(sanitizing_sample_count_list, st_return_mean_list-st_return_std_list, st_return_mean_list+st_return_std_list, facecolor='orange', alpha=0.25)
plt.fill_between(sanitizing_sample_count_list, original_sample_count_list-original_std_list, original_sample_count_list+original_std_list, facecolor='green', alpha=0.25)
print(st_return_mean_list)
print(st_return_std_list)

plt.xlabel('Clean sanitization samples (n)', fontsize=25)
plt.ylabel('Average empirical value', fontsize=25)
plt.xticks(sanitizing_sample_count_list, rotation=30, fontsize=21)  # Adjusted font size for x-axis ticks
plt.yticks(fontsize=22)  # Adjusted font size for y-axis ticks

xticks = ax.xaxis.get_major_ticks()

for i in [5, 6, 7, 9, 10, 11]:
    xticks[i].set_visible(False)

plt.grid()
plt.legend(loc='center left', fontsize=19)  # Adjusted legend fontsize
plt.tight_layout()
plt.savefig('performance_breakout.pdf')