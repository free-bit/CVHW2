#!/usr/bin/env python3

# Standard library imports
import os
import time

# Related third party imports
from cyvlfeat.sift import dsift, sift
from cyvlfeat.kmeans import kmeans # TODO: tmp
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor # TODO: tmp

# Local imports
from the2utils import *

def sample_files(files, n):
    """Shuffle the list and return n random values"""
    if (n >= len(files)):
        return files
    shuffled = np.random.shuffle(files)
    choices = np.random.choice(files, n, replace=False)
    return choices

def get_file_paths(folder, n):
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

# Find k-Nearest BoVWs from the database for a particular query
def find_k_nearest(query_bovw, other_bovws, k):
    distances = euclidean_distance(query_bovw, other_bovws, 1)
    k_indices = np.argsort(distances)[:k]
    return distances[k_indices]

def save_file(path_to_save, sample_names, bovws, slices):
    with open(path_to_save, "w+") as file:
        for i in range(len(sample_names)):
            np.savetxt(file, bovws[slices[i]:slices[i+1]], header=sample_names[i])

def read_file():
    pass

def show_results():
    pass

def find_bovws(descrs, vocab):
    """Find BoVW representation of descriptors from one or more image"""
    k = vocab.n_clusters
    cluster_indices = vocab.predict(descrs)
    # Get bincount for each row (i.e. every sample) in the matrix
    bovws = np.apply_along_axis(lambda x: np.bincount(x, minlength=k), 1, cluster_indices)
    return bovws

def perform_query(args):
    pass
    # Get descriptor of query image
    # Get BoVW of query image
    # Find k nearest neighbours

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
    if (PIPE_MODE["extract"]):
        # Get file paths
        file_paths = get_file_paths(args.readfolder, args.filecount)
        # Extract descriptors
        descrs, slices = get_descriptors(file_paths)
        # Build vocabulary
        vocab = build_vocab(descrs, k)
        # Get BoVW representation of all sample images
        find_bovws(descrs, vocab)
        if (args.save):
            pass # TODO:
            # save_file()
    if (PIPE_MODE["query"]):
        perform_query(args)
        # Show results
        if args.show:
            show_results()

def main():
    args = arg_handler()
    print(args)
    # execute_pipeline(args)

if __name__ == "__main__":
    main()