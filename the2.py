#!/usr/bin/env python3

# Standard library imports
from collections import Counter
import os
import time

# Related third party imports
from cyvlfeat.sift import dsift, sift
# from cyvlfeat.kmeans import kmeans # TODO: tmp
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsRegressor # TODO: tmp

# Local imports
from the2utils import *

# Constants & global variables
ARGS = None
GIVE_WARNING = True

# TESTED
# Per item function to find label
rxp_label = r'^.*/(.*)/\d*\.\w*$'
parse_label = lambda path : re.search(rxp_label, path).group(1)

# Vectorized version
vectorized_parse_label = np.vectorize(parse_label)

parse_name = lambda path : path[path.rfind("/")+1:]

# Vectorized version
vectorized_parse_name = np.vectorize(parse_name)


# TESTED
def sample_files(files, n):
    """Shuffle the list and return n random choices as list"""
    total = len(files)
    # If n is not provided assume all files
    if (not ARGS.fold) and (not n or (n >= total)):
        print("WARNING: Using all files without shuffling\n")
        return files
    # BUT in case of cross validation shuffle all files.
    elif ARGS.fold and (not n or n > total):
        print("WARNING: Using all files with shuffling\n")
        n = total
    # Perform shuffling in-place
    np.random.shuffle(files)
    choices = np.random.choice(files, n, replace=False)
    return list(choices)

def sample_descriptors(descrs):
    global GIVE_WARNING
    corners = ARGS.corners
    n_rows = descrs.shape[0]
    if (not corners or corners >= n_rows):
        if(GIVE_WARNING):
            print("\nWARNING: Using all descriptors ({}) extracted without shuffling\n"
                  .format(n_rows))
            GIVE_WARNING = False
        return descrs
    np.random.shuffle(descrs) # WARNING: Might be inefficient!
    sample_indices = np.random.choice(n_rows, corners, replace=False)
    return descrs[sample_indices]

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
        step = kwargs.get('step', 1)
        # print("Using dsift with fast:", fast)
        frames, descrs = dsift(image2D, step=step, fast=fast) # Might be useful: verbose=False
    else:
        # print("Using regular sift")
        frames, descrs = sift(image2D, compute_descriptor=True) # Might be useful: verbose=False
    # For debugging purposes
    # if(np.any(np.isnan(descrs))):
    #     print(descrs)
    #     input("NaN detected, proceed?")
    # if(np.any(np.isinf(descrs))):
    #     print(descrs)
    #     input("inf detected, proceed?")
    return sample_descriptors(descrs)

def get_descriptors(file_names, **kwargs):
    """
    Read n random files per folder and extract their descriptors
    Return descriptors and slices (to maintain descriptor-owner relation)
    """
    t1 = time.time()
    descrs = np.empty(shape=(0, 128), dtype='Float32')
    total = len(file_names)
    message = "Extracting features:\n"
    separator = "-" * (len(message)-1) + "\n"
    info = "Total number of files to be processed: {}\n\nProgress:".format(total)
    print(message + separator + info)
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
            print("- Processed: {}% ({}/{})".format(value, count, total))
    t2 = time.time()
    message = "\nFeature extraction completed. Time elapsed: {:.4f}\n".format(t2-t1)
    print(message + separator)
    return (descrs, slices)

# CHECKED
def find_k_nearest(query_bovws, other_bovws, k):
    """Find indices of images with k-Nearest BoVWs from the database for one or more queries"""
    print("Finding {}-nearest neighbour for {} queries among {} training files"
          .format(k, query_bovws.shape[0], other_bovws.shape[0]))
    distances = np.apply_along_axis(lambda x: euclidean_distance(x, other_bovws, 1), 1, query_bovws)
    k_indices = np.apply_along_axis(np.argsort, 1, distances)[:, :k]
    # Select values in given indices row by row
    selected_dist = np.empty((0, k))
    for i in range(distances.shape[0]):
        selected_dist = np.vstack((selected_dist, distances[i, k_indices[i]]))
    print("Done.\n")
    return k_indices, selected_dist

# TESTED
def find_accuracy(test_labels, train_labels, pred_indices):
    """
    Get ground truth and test labels.
    Perform majority voting by using labels indicated by pred_indices to decide the label.
    In case of a tie get the minimum distanced label. (Counter class enables this behaviour.)
    Calculate and return accuracy with predicted labels.
    """
    print("Calculating accuracy for {} queries and {} training files"
          .format(len(test_labels), len(train_labels)))
    true_labels = vectorized_parse_label(test_labels)
    all_train_labels = vectorized_parse_label(train_labels)
    n_row, n_col = pred_indices.shape
    pred_labels = np.empty((0, n_col))
    for i in range(n_row):
        pred_labels = np.vstack((pred_labels, all_train_labels[pred_indices[i]]))
    counters = np.apply_along_axis(Counter, 1, pred_labels)
    top_voted = [counted.most_common(1)[0][0] for counted in counters]
    labels = np.unique(all_train_labels)
    # print("All:", labels)
    # print("True:", true_labels)
    # print("Predicted:", top_voted)
    c_mat = confusion_matrix(true_labels, top_voted, labels=labels)
    accuracy = np.trace(c_mat) / np.sum(c_mat)
    np.set_printoptions(precision=2)
    if (ARGS.cmshow):
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(c_mat, labels, title='Confusion matrix')
        plt.show()
    print("Score achieved: {}".format(accuracy))
    return accuracy, top_voted

def find_bovws(descrs, slices, vocab):
    """Find BoVW representation of descriptors from one or more image"""
    print("Finding BoVWs...")
    k = vocab.n_clusters
    # Get cluster index of each row descriptor vector and convert it to a row vector
    cluster_indices = vocab.predict(descrs)
    # Each bovw row vector will have k columns
    bovws = np.empty(shape=(0, k))
    for i in range(len(slices)-1):
        ci = cluster_indices[slices[i] : slices[i+1]]
        bovw = np.bincount(ci, minlength=k)
        bovws = np.vstack((bovws, bovw))
    print("BoVWs are constructed.\n")
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
    print(("Based on features with shape:{} "+
           "building vocabulary with size: {}").format(descrs.shape, k))
    clusters = KMeans(n_clusters=k).fit(descrs) # sample_weight, get_params
    # TODO: inverted indices might be removed but requires update on read and save!
    inv_indices = [None] * k
    for c_index in range(k):
        inv_indices[c_index] = np.where(clusters.labels_ == c_index)
    clusters.inv_indices_ = inv_indices
    # Remove unused variable
    del clusters.labels_
    print("Vocabulary is constructed.\n")
    return clusters

def create_file_name(type, n_files):
    ext_method = "sift"
    if (ARGS.dense):
        ext_method = "d" + ext_method + "_step{}".format(ARGS.stepsize)
        if (ARGS.fast):
            ext_method = "f" + ext_method
    name = "{}_{}_corners{}_cluster{}_files{}.txt".format(type, ext_method, ARGS.corners, 
                                                          ARGS.clusters, n_files)
    return name

def train(train_labels, train_descrs, train_slices):
    # Build vocabulary from training images
    vocab = build_vocab(train_descrs, ARGS.clusters)
    # Get BoVW representation of all sampled training images
    train_bovws = find_bovws(train_descrs, train_slices, vocab)
    # Save results (if specified)
    if (ARGS.save):
        # Save bovw
        bovw_name = create_file_name("bovws", train_bovws.shape[0])
        save_bovws(bovw_name, train_labels, train_bovws)
        # Save vocab
        vocab_name = create_file_name("vocab", train_bovws.shape[0])
        save_vocab(vocab_name, vocab)
        
    return train_bovws, vocab

def test(test_labels, test_descrs, test_slices, train_labels, train_bovws, vocab):
    # Get BoVW representation of test images
    test_bovws = find_bovws(test_descrs, test_slices, vocab)
    # Find k nearest neighbours
    pred_indices, selected_dist = find_k_nearest(test_bovws, train_bovws, ARGS.knn)
    # Calculate accuracy
    score, top_voted = find_accuracy(test_labels, train_labels, pred_indices)
    # Save predictions
    if (ARGS.save):
        ARGS.clusters = vocab.n_clusters
        result_name = create_file_name("results_knn{}".format(ARGS.knn), train_bovws.shape[0])
        test_names = vectorized_parse_name(test_labels)
        save_pred_labels(result_name, test_names, top_voted)
    return score

# Distribute descriptors of each file into their folds
def split_descrs(descrs, slices, fold_count):
    split_count = len(slices) - 1 
    splits = [None] * split_count
    for i in range(split_count):
        splits[i] = descrs[slices[i] : slices[i+1]]
    splits = np.array_split(splits, fold_count)
    return splits

def setup_descrs_slices(descr_folds, test_fold_index):
    """
    Partition descr_folds based on the index of selected test fold.
    For descr_folds structure see 'print_descr_fold_structure'
    Returns: 
    train_descrs, test_descrs: Descriptors to be used for training and test
    train_slices, test_slices: Slices indicating boundaries of descriptors for each file
    """
    print("Setting up necessary variables for testing with fold-{}".format(test_fold_index))
    fold_count = len(descr_folds)
    # Grab test fold
    test_fold = descr_folds[test_fold_index]
    test_slices = [0]
    test_descrs = np.empty((0, 128))
    for each_group in test_fold:
        test_slices.append(test_slices[-1] + each_group.shape[0])
        test_descrs = np.vstack((test_descrs, each_group))

    # Rest are training folds
    train_folds = descr_folds[:test_fold_index] + descr_folds[test_fold_index+1:]
    train_descrs = np.empty((0, 128))
    train_slices = [0]
    for train_fold in train_folds:
        for each_group in train_fold:
            train_slices.append(train_slices[-1] + each_group.shape[0])        
            train_descrs = np.vstack((train_descrs, each_group))
    print("Setup completed for testing with fold-{}\n".format(test_fold_index))
    # print(test_slices) #TODO
    # print(test_descrs.shape) #TODO
    # print(train_slices) #TODO
    # print(train_descrs.shape) #TODO
    return train_descrs, train_slices, test_descrs, test_slices

def print_descr_fold_structure(descr_folds):
    """
    Structure of descr_folds is as follows:
    ---------------------------------------
    type(descr_folds) -> list: Folds
    len(descr_folds) -> k: Number of folds
    
    type(descr_folds[i]) -> np.array: Groups of descriptors 
    descr_folds[i].shape -> (n,): Number of descriptor groups in the fold
    
    type(descr_folds[i][j] -> np.array: One group of descriptors 
    descr_folds[i][j].shape -> (x, 128): Each group of descriptors belongs to a file
    """
    fold_count = len(descr_folds)
    for i in range(fold_count):
        group_count = descr_folds[i].shape[0]
        message = "\nFold-{}\n".format(i)
        separator = "-" * (len(message)-1)
        info = "\nNumber of groups under this fold: {}\n".format(group_count)
        print(message + separator + info)
        for j in range(group_count):
            print("Group-{}:".format(j))
            print(descr_folds[i][j].shape,"\n")
        print(separator)

def x_validate(labels, descrs, slices):
    # Get folds
    fold_count = ARGS.fold
    label_folds = np.array_split(labels, fold_count)
    descr_folds = split_descrs(descrs, slices, fold_count)

    # For testing purposes
    # print_descr_fold_structure(descr_folds) #TODO:
    
    # Keep score for each iteration
    scores = np.empty(fold_count)

    # In each iteration pick one fold as test and the rest as training data 
    for i in range(fold_count):
        # Setup parameters with respect to current fold
        train_descrs, train_slices, test_descrs, test_slices = setup_descrs_slices(descr_folds, i)
        
        # Build model on training folds
        train_labels = np.concatenate((*label_folds[:i], *label_folds[i+1:]))
        test_labels = label_folds[i]

        # Train on train folds
        train_bovws, vocab = train(train_labels, train_descrs, train_slices)
        
        # Test on test folds
        message = "\nTesting with cross validation on test fold-{}:\n".format(i)
        separator = "-" * (len(message)-2)
        print(message + separator)
        
        # Calculate score
        scores[i] = test(test_labels, test_descrs, test_slices, train_labels, train_bovws, vocab)
        print(separator + "\n")

    print("Average score:", np.mean(scores))

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
        descrs, slices = get_descriptors(train_labels, dense=ARGS.dense, fast=ARGS.fast,
                                         step=ARGS.stepsize, percent=ARGS.percent)
        # If cross validation mode enabled
        if (PIPE_MODE["x_valid"]):
            message = "Training with cross validation on data under {}:\n".format(ARGS.trainfolder)
            separator = "-" * (len(message)-2)
            print(message + separator)
            x_validate(train_labels, descrs, slices)
            print(separator)
        # Otherwise perform training without folds
        else:
            message = "\nTraining on data under {}:\n".format(ARGS.trainfolder)
            separator = "-" * (len(message)-2)
            print(message + separator)
            train_bovws, vocab = train(train_labels, descrs, slices)
            print(separator)

    # If test mode enabled
    if (PIPE_MODE["test"]):
        read_file = (train_labels is None) or (train_bovws is None) or (vocab is None)
        if (read_file):
            print("Reading BoVW information from file: {}".format(ARGS.bovwfile))
            train_labels, train_bovws = read_bovws(ARGS.bovwfile)
            print("Reading vocabulary information from file: {}".format(ARGS.vocabfile))
            vocab = read_vocab(ARGS.vocabfile)
        message = "\nTesting on data under {}:\n".format(ARGS.testfolder)
        separator = "-" * (len(message)-2)
        print(message + separator + "\n")
        # Extract descriptors of test images
        test_labels = get_file_paths(ARGS.testfolder)
        # Extract descriptors of training images
        descrs, slices = get_descriptors(test_labels, dense=ARGS.dense, fast=ARGS.fast, 
                                         step=ARGS.stepsize, percent=ARGS.percent)
        test(test_labels, descrs, slices, train_labels, train_bovws, vocab)
        print(separator)

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
    separator = "-" * (len(message)-1)
    message = message + separator + "\n" +\
              ("1) Train: {}\n\n" +
               "   - CV : {}\n\n" +
               "2) Test : {}\n").format(s1, s2, s3) +\
              separator + "\n"
    print(message)

def main():
    global ARGS
    ARGS = arg_handler()
    # If required args are parsed properly
    if ARGS:
        show_current_config()
        show_pipe()
        execute_pipeline()

if __name__ == "__main__":
    main()