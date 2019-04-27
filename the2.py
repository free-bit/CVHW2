#!/usr/bin/env python3

# Standard library imports
import os
import re
import time

# Related third party imports
from cyvlfeat.sift import dsift, sift
from cyvlfeat.kmeans import kmeans # TODO: tmp
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor # TODO: tmp

# Local imports
from the2utils import *

# Tested
def sample_files(files, n):
    """Shuffle the list and return n random values"""
    if (not n or (n >= len(files))):
        return files
    shuffled = np.random.shuffle(files)
    choices = np.random.choice(files, n, replace=False)
    return choices

# Tested
def get_file_paths(folder, n=None):
    """
    Find all files under given path in the form of: 
    folder
    - folder1: 
            - images
    ...
    - folderN:
            - images
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
    print("Total number of files to be processed:", len(file_names))
    slices = [0]
    for file in file_names:
        _, gray = read_image(file)
        extracted = get_SIFT_descriptor(gray, **kwargs)
        descrs = np.vstack((descrs, extracted))
        slices.append(descrs.shape[0])
    t2 = time.time()
    print("Time elapsed:", t2-t1)
    return (descrs, slices)

# Tested
def find_k_nearest(query_bovw, other_bovws, k):
    """Find k-Nearest BoVWs from the database for a particular query"""
    distances = euclidean_distance(query_bovw, other_bovws, 1)
    k_indices = np.argsort(distances)[:k]
    return distances[k_indices]

# Tested
def save_file(path_to_save, sample_names, bovws, slices):
    with open(path_to_save, "w+") as file:
        for i in range(len(sample_names)):
            np.savetxt(file, bovws[slices[i]:slices[i+1]], header=sample_names[i])

# TODO: Incomplete
def read_file(path_to_read):
    rxp_varname = r'^(?:# (.*))$'
    var_names = []
    values = []
    with open(path_to_read, "r") as file:
        index = -1
        line = file.readline()
        while (line):
            if (line[0] == "#"):
                var_names.append(re.findall(rxp_varname, line))
                values.append(str())
                index += 1
            else:
                values[index] += line
            line = file.readline()
    print(var_names)
    for value in values:
        value = np.fromstring(value, sep=' ')
        print(value)

def show_results():
    pass

def find_bovws(descrs, vocab):
    """Find BoVW representation of descriptors from one or more image"""
    k = vocab.n_clusters
    cluster_indices = vocab.predict(descrs)
    # Get bincount for each row (i.e. every sample) in the matrix
    bovws = np.apply_along_axis(lambda x: np.bincount(x, minlength=k), 1, cluster_indices)
    return bovws

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
    return clusters

def execute_pipeline(args):
    global PIPE_MODE 
    k=5 # TODO: hyperparameter from input file
    if (PIPE_MODE["train"]):
        # Get file paths
        file_paths = get_file_paths(args.trainfolder, args.filecount)
        # Extract descriptors of training images
        descrs, slices = get_descriptors(file_paths)
        # Build vocabulary from training images
        vocab = build_vocab(descrs, k)
        # Get BoVW representation of all sampled training images
        train_bovws = find_bovws(descrs, vocab)
        if (args.save):
            pass # TODO:
            # save_file()
    if (PIPE_MODE["test"]):
        # Extract descriptors of test images
        file_paths = get_file_paths(args.testfolder)
        # Extract descriptors of test images
        descrs, slices = get_descriptors(file_paths)
        # Get BoVW representation of test images
        test_bovws = find_bovws(descrs, vocab)
        # Find k nearest neighbours
        find_k_nearest(test_bovws, train_bovws, k)
        # Show results
        if args.show:
            show_results()

def main():
    args = arg_handler()
    if args:
        print(args)
        # execute_pipeline(args)
    # Debugging
    else:
        read_file("test.txt")

if __name__ == "__main__":
    main()