#!/usr/bin/env python3

# Standard library imports
from collections import Counter
import os
import re
import time

# Related third party imports
from cyvlfeat.sift import dsift, sift
from cyvlfeat.kmeans import kmeans # TODO: tmp
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor # TODO: tmp
from sklearn.metrics import accuracy_score

# Local imports
from the2utils import *

# Global variables
ARGS = None

# TESTED
# Per item function to find label
rxp_label = r'^.*/(.*)/\d*\.\w*$'
parse_label = lambda path : re.search(rxp_label, path).group(1)

# Vectorized version
vectorized_parse_label = np.vectorize(parse_label)

# TESTED
def sample_files(files, n):
    """Shuffle the list and return n random choices as list"""
    if (not n or (n >= len(files))):
        return files
    # Perform shuffling in-place
    np.random.shuffle(files)
    choices = np.random.choice(files, n, replace=False)
    return list(choices)

# TESTED
def get_file_paths(folder, n=None):
    """
    Find all files under given path in the form of: 
    folder
    - folder1: 
            - images
    ...
    - folderN:
            - images
    Returns list of file paths
    """
    file_paths = []
    subfolders = os.listdir(folder)
    for subfolder in subfolders:
        subfolder = folder + subfolder + "/"
        files = os.listdir(subfolder)
        file_paths.extend([(subfolder + file) for file in files])
    samples = sample_files(file_paths, n)
    return samples

def get_SIFT_descriptor(image2D, **kwargs):
    """
    Perform SIFT on a single image.
    frames[i] -> (x,y)
    descrs[i] -> (f1,f2,f3...f128)
    Configure SIFT algorithm with followings:
    dense: Apply d-SIFT (Default: True)
    float: Descriptor are returned in floating point (Default: False)
    fast: Fast approximation for d-SIFT (Default: False)
    Returns np.array
    """
    dense = kwargs.get('dense', True)
    float = kwargs.get('float', False)
    frames = descrs = None
    if dense:
        fast = kwargs.get('fast', False)
        frames, descrs = dsift(image2D, fast=fast, float_descriptors=float)
    else:
        frames, descrs = sift(image2D, compute_descriptor=True, float_descriptors=float)
    return descrs

def get_descriptors(file_names, **kwargs):
    """
    Read n random files per folder and extract their descriptors
    Return descriptors and slices (to maintain descriptor-owner relation)
    """
    t1 = time.time()
    descrs = np.empty(shape=(0, 128))
    total = len(file_names)
    print("Total number of files to be processed:", total)
    slices = [0]
    # Use for percentage calculations
    percentage = kwargs.get('percentage', 10)
    count = 0
    step = total / percentage
    for file in file_names:
        _, gray = read_image(file, read_color=False)
        extracted = get_SIFT_descriptor(gray, **kwargs)
        descrs = np.vstack((descrs, extracted))
        slices.append(descrs.shape[0])
        # Notify every X%
        count += 1
        if (count % step == 0):
            print("{}%".format(count*percentage/step))
    t2 = time.time()
    print("Completed. Time elapsed:", t2-t1)
    return (descrs, slices)

# TESTED
def find_k_nearest(query_bovws, other_bovws, k):
    """Find indices of images with k-Nearest BoVWs from the database for one or more queries"""
    distances = np.apply_along_axis(lambda x: euclidean_distance(x, other_bovws, 1), 1, query_bovws)
    k_indices = np.apply_along_axis(np.argsort, 1, distances)[:, :k]
    # Select values in given indices row by row
    selected_dist = np.empty((0, k))
    for i in range(distances.shape[0]):
        selected_dist = np.vstack((selected_dist, distances[i, k_indices[i]]))
    return k_indices, selected_dist

# TESTED
def find_accuracy(test_paths, train_paths, indices):
    """
    Get ground truth and test labels.
    Perform majority voting by using labels indicated by indices to decide the label.
    In case of a tie get the minimum distanced label. (Counter class enables this behaviour.)
    Calculate and return accuracy.
    """
    true_labels = vectorized_parse_label(test_paths)
    all_train_labels = vectorized_parse_label(train_paths)
    n_row, n_col = indices.shape
    pred_labels = np.empty((0, n_col))
    for i in range(n_row):
        pred_labels = np.vstack((pred_labels, all_train_labels[indices[i]]))
    counters = np.apply_along_axis(Counter, 1, pred_labels)
    top_voted = [counted.most_common(1)[0][0] for counted in counters]
    accuracy = accuracy_score(true_labels, top_voted)
    return accuracy

# TESTED
def save_bovws(path_to_save, file_names, bovws):
    with open(path_to_save, "w+") as file:
        for i in range(len(file_names)):
            np.savetxt(file, bovws[i], header=file_names[i])
# TESTED
def save_centers(path_to_save, centers):
    with open(path_to_save, "w+") as file:
        header = "Cluster centers:"
        np.savetxt(file, centers, header=header)
# TESTED
def save_inv_indices(path_to_save, inv_indices):
    with open(path_to_save, "w+") as file:
        header = "# Inverted indices:\n"
        file.write(header)
        for cluster in inv_indices:
            np.savetxt(file, cluster, fmt="%d")
# TESTED
def read_bovws(path_to_read):
    """Read a file containing descriptors, returns name and matrix pairs as dict"""
    rxp_file_name = r'^(?:# (.*))$'
    file_names = []
    bovws = []
    with open(path_to_read, "r") as file:
        line = file.readline()
        bovw = None
        while (line):
            if (line[0] == "#"):
                file_name = re.search(rxp_file_name, line).group(1)
                file_names.append(file_name)
                bovws.append(str())
            else:
                bovws[-1] += line
            line = file.readline()
    # Convert strings to numpy arrays
    for i in range(len(bovws)):
        bovws[i] = np.fromstring(bovws[i], sep='\n')
    bovws = np.array(bovws)
    return file_names, bovws

def show_results():
    pass

def find_bovws(descrs, slices, vocab):
    """Find BoVW representation of descriptors from one or more image"""
    k = vocab.n_clusters
    # Get cluster index of each row descriptor vector and convert it to a row vector
    cluster_indices = vocab.predict(descrs)
    # Each bovw row vector will have k columns
    bovws = np.empty(shape=(0, k))
    for i in range(len(slices)-1):
        ci = cluster_indices[slices[i] : slices[i+1]]
        bovw = np.bincount(ci, minlength=k)
        bovws = np.vstack((bovws, bovw))
    # Normalize each row vector
    return l1_normalize(bovws)

def build_vocab(descrs, k):
    """
    Compute k-means
    Return KMeans instance including:
    n_clusters:       number of clusters, k
    cluster_centers_: cluster indices of each cluster
    clusters.labels_: cluster indices of each sample
    inv_indices_:     inverted index to be used in look up
    """
    clusters = KMeans(n_clusters=k).fit(descrs) # sample_weight, get_params
    inv_indices = [None] * k
    for c_index in range(k):
        inv_indices[c_index] = np.where(clusters.labels_ == c_index)
    clusters.inv_indices_ = inv_indices
    # TODO: Remove unused variable
    # del clusters.labels_
    return clusters

def execute_pipeline():
    global PIPE_MODE 
    if (PIPE_MODE["train"]):
        # Get file paths (select c-many files randomly)
        print("Getting file paths (select c-many files randomly)") #TODO: remove
        file_paths = get_file_paths(ARGS.trainfolder, ARGS.filecount)
        print("Count:", len(file_paths)) #TODO: remove
        input("Proceed?\n") #TODO: remove
        
        # Extract descriptors of training images
        print("Extracting descriptors of training images") #TODO: remove
        descrs, slices = get_descriptors(file_paths)
        print("Descrs size:", descrs.shape) #TODO: remove
        print("Slices size:", len(slices)) #TODO: remove
        input("Proceed?\n") #TODO: remove
        
        # Build vocabulary from training images
        vocab = build_vocab(descrs, ARGS.clusters)
        print("Vocab size: ", vocab.n_clusters) #TODO: remove
        input("Proceed?\n") #TODO: remove

        # Get BoVW representation of all sampled training images
        train_bovws = find_bovws(descrs, slices, vocab)
        print("bovws size: ", train_bovws.shape) #TODO: remove
        input("Proceed?\n") #TODO: remove

        # Save results
        if (ARGS.save):
            save_bovws("bovws.txt", file_paths, train_bovws)
            save_centers("centers.txt", vocab.cluster_centers_)
            save_inv_indices("indices.txt", vocab.inv_indices_)

    # TODO: Check later
    if (PIPE_MODE["test"]):
        # Extract descriptors of test images
        file_paths = get_file_paths(ARGS.testfolder)
        # Extract descriptors of test images
        descrs, slices = get_descriptors(file_paths)
        # Get BoVW representation of test images
        test_bovws = find_bovws(descrs, vocab)
        # Find k nearest neighbours
        find_k_nearest(test_bovws, train_bovws, k)
        # Show results
        if ARGS.show:
            show_results()

def get_current_config():
    """Return a string indicating current parameter configuration"""
    config = vars(ARGS)
    message = "\nRunning with the following parameter settings:\n"
    separator = "-" * len(message) + "\n"
    lines = ""
    for item, key in config.items():
        lines += "{}: {}\n".format(item, key)
    return (message + separator + lines + separator) 

def show_current_config():
    """Print current parameter configuration"""
    print(get_current_config())

def main():
    global ARGS
    ARGS = arg_handler()
    # If required args are parsed properly
    if ARGS:
        show_current_config()
        execute_pipeline()
    # Debugging
    else:
        read_bovws("test.txt")

if __name__ == "__main__":
    main()