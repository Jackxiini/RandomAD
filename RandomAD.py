import numpy as np
import time
import argparse
import os
from sklearn.exceptions import UndefinedMetricWarning
#from utils import data_profile, data_sets_names, data_set_keys
from randomad_core import (
    KNN,
    KNN_norm,
    MiniRocket_fit,
    RandomAD_mask,
    RandomAD_run,
)

from scipy.signal import find_peaks, correlate

argparser = argparse.ArgumentParser()
argparser.add_argument('--n_kernel', type=int, default=1000)
argparser.add_argument('--alpha', type=float, default=0.5)
argparser.add_argument('--beta', type=float, default=0.5)
argparser.add_argument('--rate', type=float, default=0.5)
argparser.add_argument('--dataset', type=str, default='UCR')
args = argparser.parse_args()

dataset = args.dataset
dataset_idx = 0
k_neighbors = 3
model = 'RandomAD'

def calculate_window_size(data):
    autocor = correlate(data, data, mode='full')
    autocor = autocor[len(autocor)//2:]

    peak_range = np.arange(10, min(len(autocor), 1001))
    peaks, _ = find_peaks(autocor[peak_range], distance=10)
    peaks = peak_range[peaks]
    
    if len(peaks) > 0:
        m = peaks[0]
    else:
        m = 80
    
    return int(np.floor(m))

def select_window_sizes(upper_bound, lower_bound=10, num_candidates=4):
    # Ensure the upper bound is greater than the lower bound
    if upper_bound <= lower_bound:
        raise ValueError("Upper bound must be greater than lower bound.")
    
    # Calculate evenly spaced candidates within the range
    window_sizes = np.linspace(lower_bound, upper_bound, num_candidates, dtype=int)
    
    return window_sizes

def select_incremental_window_sizes(upper_bound, lower_bound=10, num_candidates=4):
    increment = (upper_bound - lower_bound) // (num_candidates - 1)
    return np.unique([lower_bound + i * increment for i in range(num_candidates)])

def select_best_window_size(point_anomaly_scores_list, window_sizes):
    max_differences = []
    
    for i, scores in enumerate(point_anomaly_scores_list):
        max_score = np.max(scores)
        max_index = np.argmax(scores)
        exclusion_start = max(0, max_index - window_sizes[i])
        exclusion_end = min(len(scores), max_index + window_sizes[i])
        masked_scores = np.copy(scores)
        masked_scores[exclusion_start:exclusion_end] = -np.inf
        second_max_score = np.max(masked_scores)
        score_difference = max_score - second_max_score
        max_differences.append(score_difference)

    best_index = np.argmax(max_differences)
    best_window_size = window_sizes[best_index]
    anomaly_scores = point_anomaly_scores_list[best_index]
    
    return best_window_size, anomaly_scores

def detect_delimiter(file_path):
    with open(file_path, 'r') as file:
        line = file.readline()
        if ',' in line:
            return ','
        elif '\t' in line:
            return '\t'
        else:
            return ' '
        
if dataset == 'UCR':
    dataset_folder = "data/UCR"
    file_list = os.listdir(dataset_folder)
elif dataset == 'custom':
    pass
               
accuracy_scores = 0

# results = []
start_time = time.time()
filename = file_list[dataset_idx]
print(filename)

if dataset == 'UCR':
    if not filename.endswith('.txt'): pass
    vals = filename.split('.')[0].split('_')
    dnum, vals = int(vals[0]), vals[-3:]
    vals = [int(i) for i in vals]
    temp = np.loadtxt(dataset_folder + '/' + filename)
    min_temp, max_temp = np.min(temp), np.max(temp)
    temp = (temp - min_temp) / (max_temp - min_temp)
    train, test = temp[:vals[0]], temp[vals[0]:]
    labels = np.zeros_like(test)
    labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
    train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)

data = np.concatenate([train, test], axis=0)

window_size = calculate_window_size(train.flatten())
window_sizes = select_incremental_window_sizes(upper_bound=window_size, lower_bound=10, num_candidates=4)

#window_sizes = select_incremental_window_sizes(upper_bound=200, lower_bound=10, num_candidates=4)
print('Window Sizes:', window_sizes)

# Segment time series into subsequences
def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)

anomaly_scores_list = []
for window_size in window_sizes:

    train_windows = create_windows(train, window_size)
    test_windows = create_windows(test, window_size)
    train_windows = train_windows.reshape(-1, window_size)
    test_windows = test_windows.reshape(-1, window_size)

    if model == 'KNN':
        train_distances = KNN(k_neighbors, train_windows, train_windows)
        test_distances = KNN(k_neighbors, train_windows, test_windows)
    elif model == 'KNN_norm':
        train_distances = KNN_norm(k_neighbors, train_windows, train_windows)
        test_distances = KNN_norm(k_neighbors, train_windows, test_windows)
    elif model == 'RandomAD':
        num_features = args.n_kernel
        use_pca = False
        pca_variance_ratio = 0.95
        para = MiniRocket_fit(train_windows, num_features=num_features)
        mask = RandomAD_mask(para, train_windows, keep_features_ratio=args.rate, num_samples=100, alpha=args.alpha, beta=args.beta)
        train_distances = RandomAD_run(para, mask, k_neighbors, train_windows, use_pca=use_pca)
        test_distances = RandomAD_run(para, mask, k_neighbors, train_windows, test_windows, use_pca=use_pca)

    train_distances = np.concatenate([np.zeros(window_size-1), train_distances])
    test_distances = np.concatenate([np.zeros(window_size-1), test_distances])
    anomaly_scores = np.concatenate([train_distances, test_distances])
    anomaly_scores_list.append(anomaly_scores)

best_window_size, anomaly_scores = select_best_window_size(anomaly_scores_list, window_sizes)
print('Best Window Size:', best_window_size)

if dataset == 'UCR':
    true_anomaly_start = vals[1]
    true_anomaly_end = vals[2]
    true_anomaly_starts = [vals[1]]
    true_anomaly_ends = [vals[2]]

elif dataset == 'custom':
    pass

test_labels = labels[:-window_size+1]

init_score = train_distances
test_score = test_distances

anomaly_location = np.argmax(anomaly_scores)
L = max(100, true_anomaly_end - true_anomaly_start)
is_anomaly_correct = 0
if true_anomaly_start - L <= anomaly_location <= true_anomaly_end + L:
    accuracy_scores += 1
    is_anomaly_correct = 1

print("Time taken:", round(time.time() - start_time, 2), "seconds")
print("Kernel Number:", args.n_kernel)
print("Kernel selection rate:", args.rate)
print("Accuracy Score:", accuracy_scores)  # 1 for correct, 0 for incorrect