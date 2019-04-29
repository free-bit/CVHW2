#!/usr/bin/env python3

# Standard library imports
from collections import Counter
import os
import time

# Related third party imports
from cyvlfeat.sift import dsift, sift
from cyvlfeat.kmeans import kmeans # TODO: tmp
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor # TODO: tmp

# Local imports
from the2utils import *

# Constants & global variables
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
    frames = descrs = None
    if dense:
        fast = kwargs.get('fast', False)
        # print("Using dsift with fast:", fast)
        frames, descrs = dsift(image2D, fast=fast) # Might be useful: verbose=False
    else:
        # print("Using regular sift")
        frames, descrs = sift(image2D, compute_descriptor=True) # Might be useful: verbose=False
    # TODO: debugging
    if(np.any(np.isnan(descrs))):
        print(descrs)
        input("NaN detected, proceed?")
    if(np.any(np.isinf(descrs))):
        print(descrs)
        input("inf detected, proceed?")
    return descrs

def get_descriptors(file_names, **kwargs):
    """
    Read n random files per folder and extract their descriptors
    Return descriptors and slices (to maintain descriptor-owner relation)
    """
    t1 = time.time()
    descrs = np.empty(shape=(0, 128), dtype='Float32')
    total = len(file_names)
    print("Extracting features:")
    print("Total number of files to be processed:", total)
    slices = [0]
    # Use for percentage calculations
    percent = kwargs.get('percent', 10)
    count = 0
    step = (total * percent) / 100
    for file in file_names:
        _, gray = read_image(file, read_color=False)
        extracted = get_SIFT_descriptor(gray, **kwargs)
        descrs = np.vstack((descrs, extracted))
        slices.append(descrs.shape[0])
        # Notify every X%
        count += 1
        if (count % step == 0):
            value = (count * percent) / step
            print("Processed: {}% ({}/{})".format(value, count, total))
    t2 = time.time()
    print("Feature extraction completed. Time elapsed:", t2-t1)
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
def find_accuracy(test_labels, train_labels, pred_indices):
    """
    Get ground truth and test labels.
    Perform majority voting by using labels indicated by pred_indices to decide the label.
    In case of a tie get the minimum distanced label. (Counter class enables this behaviour.)
    Calculate and return accuracy.
    """
    true_labels = vectorized_parse_label(test_labels)
    all_train_labels = vectorized_parse_label(train_labels)
    n_row, n_col = pred_indices.shape
    pred_labels = np.empty((0, n_col))
    for i in range(n_row):
        pred_labels = np.vstack((pred_labels, all_train_labels[pred_indices[i]]))
    counters = np.apply_along_axis(Counter, 1, pred_labels)
    top_voted = [counted.most_common(1)[0][0] for counted in counters]
    accuracy = accuracy_score(true_labels, top_voted)
    return accuracy

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
    print(("Based on features (shape:{}) "+
           "building vocabulary with size: {}").format(descrs.shape, k))
    clusters = KMeans(n_clusters=k).fit(descrs) # sample_weight, get_params
    inv_indices = [None] * k
    for c_index in range(k):
        inv_indices[c_index] = np.where(clusters.labels_ == c_index)
    clusters.inv_indices_ = inv_indices
    # TODO: Remove unused variable
    del clusters.labels_
    return clusters

def create_file_name(type, k, filecount):
    name = "{}_cluster{}_samples{}.txt".format(type, k, filecount)
    return name

def train(train_labels, train_descrs, train_slices):
    # Build vocabulary from training images
    vocab = build_vocab(train_descrs, ARGS.clusters)
    # Get BoVW representation of all sampled training images
    train_bovws = find_bovws(train_descrs, train_slices, vocab)
    # Save results (if specified)
    if (ARGS.save):
        # Save bovw
        bovw_name = create_file_name("bovws", ARGS.clusters, train_bovws.shape[0])
        save_bovws(bovw_name, train_labels, train_bovws)
        # Save vocab
        vocab_name = create_file_name("vocab", ARGS.clusters, train_bovws.shape[0])
        save_vocab(vocab_name, vocab)
        
    return train_bovws

def test(test_labels, test_descrs, test_slices, train_labels, train_bovws, vocab):
    # Get BoVW representation of test images
    test_bovws = find_bovws(test_descrs, test_slices, vocab)
    # Find k nearest neighbours
    pred_indices, selected_dist = find_k_nearest(test_bovws, train_bovws, k)
    # Calculate accuracy
    score = find_accuracy(test_labels, train_labels, pred_indices)
    # Show results
    if ARGS.show:
        show_results()
    return score

def x_validate(labels, descrs, slices):
    # Get folds
    label_folds = np.array_split(labels, ARGS.fold)
    descr_folds = np.array_split(descrs, ARGS.fold)
    # In each iteration pick one fold as test and the rest as training data 
    for i in range(len(folds)):
        # Build model on training folds
        train_labels = np.concatenate((*label_folds[:i], *label_folds[i+1:]))
        train_descrs = np.concatenate((*descr_folds[:i], *descr_folds[i+1:]))
        train(train_labels, train_descrs, slices)
        # Test on test folds
        test_labels = label_folds[i]
        test_descrs = descr_folds[i]
        test(test_labels, test_descrs, slices)

def execute_pipeline():
    global PIPE_MODE
    train_labels = None
    train_bovws = None
    vocab = None
    # If train mode enabled
    if (PIPE_MODE["train"]):
        # Get train file paths (select c-many files randomly)
        train_labels = get_file_paths(ARGS.trainfolder, ARGS.filecount)
        # Extract descriptors of training images
        descrs, slices = get_descriptors(train_labels, dense=ARGS.dense, 
                                         fast=ARGS.fast, percent=ARGS.percent)
        # If cross validation mode enabled
        if (PIPE_MODE["x_valid"]):
            print("Training with cross validation on data under {}".format(ARGS.trainfolder))
            x_validate(train_labels, descrs, slices)
        # Otherwise perform training without folds
        else:
            print("Training on data under {}".format(ARGS.trainfolder))
            train_bovws = train(train_labels, descrs, slices)

    # If test mode enabled
    if (PIPE_MODE["test"]):
        if (not train_labels or not train_bovws or not vocab): # TODO: vocab file to read
            print("Reading BoVW information from file: {}".format(ARGS.bovwfile))
            train_labels, train_bovws = read_bovws(ARGS.bovwfile)
            vocab = read_vocab(ARGS.vocabfile)
        print("Testing on data under {}".format(ARGS.testfolder))
        # Extract descriptors of test images
        test_labels = get_file_paths(ARGS.testfolder)
        # Extract descriptors of training images
        descrs, slices = get_descriptors(test_labels, dense=ARGS.dense, 
                                         fast=ARGS.fast, percent=ARGS.percent)
        score = test(test_labels, descrs, slices, train_labels, train_bovws, vocab)
        print("Score achieved:", score)

def get_current_config():
    global ARGS
    """Return a string indicating current parameter configuration"""
    config = vars(ARGS)
    message = "\nRunning with the following parameter settings:\n"
    separator = "-" * (len(message)-2) + "\n"
    lines = ""
    for item, key in config.items():
        lines += "- {}: {}\n".format(item, key)
    return (message + separator + lines + separator) 

def show_current_config():
    """Print current parameter configuration"""
    print(get_current_config())

def show_pipe():
    """Print pipeline modes to be executed"""
    enum = {True: "Enabled", False: "Disabled"}
    s1, s2, s3 = enum[PIPE_MODE["train"]], enum[PIPE_MODE["x_valid"]], enum[PIPE_MODE["test"]]
    message = "Active pipeline modes:\n"
    separator = "-" * (len(message)-1) + "\n"
    message = message + separator +\
              ("1) Train: {}\n\n" +
               "   - CV : {}\n\n" +
               "2) Test : {}\n").format(s1, s2, s3) +\
              separator
    print(message)

def main():
    global ARGS
    ARGS = arg_handler()
    # If required args are parsed properly
    if ARGS:
        show_current_config()
        show_pipe()
        execute_pipeline()
    # Debugging
    else:
        pass

if __name__ == "__main__":
    main()