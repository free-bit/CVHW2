#!/usr/bin/env python3

# Standard library imports
import os
import time

# Related third party imports
from cyvlfeat.sift import dsift, sift
from cyvlfeat.kmeans import kmeans

# Local imports
from the2utils import *

def apply_kNN():
    pass

def BoVW():
    pass
'''
Apply SIFT algorithm with parameters:
dense: Apply d-SIFT (Default: True)
float: Descriptor are returned in floating point (Default: False)
fast: Fast approximation for d-SIFT (Default: False)
frames[i] -> (x,y)
descrs[i] -> (f1,f2,f3...f128)    
'''
def apply_SIFT(image2D, **kwargs):
    dense = kwargs.get('dense', True)
    float = kwargs.get('float', False)
    frames = descrs = None
    if dense:
        fast = kwargs.get('fast', False)
        frames, descrs = dsift(image2D, fast=fast, float_descriptors=float)
    else:
        frames, descrs = sift(image2D, compute_descriptor=True, float_descriptors=float)
    return descrs

def get_file_paths(folder):
    file_paths = []
    subfolders = os.listdir(folder)
    for subfolder in subfolders:
        subfolder = folder + subfolder + "/"
        file_paths.extend([(subfolder + file) for file in os.listdir(subfolder)])
    return file_paths

def get_all_descriptors(folder):
    t1 = time.time()
    all_descrs = np.array([])
    files = get_file_paths(folder)
    print("Total number of files to be processed:", len(files))
    for file in files:
        _, gray = read_image(file)
        extracted = apply_SIFT(gray)
        all_descrs = np.append(all_descrs, extracted)
    t2 = time.time()
    print("Time elapsed:", t2-t1)
    return all_descrs

def apply_kmeans(feat_vecs, k, max_iter=10):
    n, m = feat_vecs.shape
    proceed = True
    centers = None
    print("Rows: {}, Cols: {}, Clusters: {}".format(n, m, k))
    while (proceed):
        proceed = False
        # TODO: Remove later: 
        # pseudo_rand = np.random.RandomState(30)
        # centers = pseudo_rand.uniform(size=(k, m))
        centers = np.random.uniform(size=(k, m))
        print("Initial centers:\n", centers)
        for _ in range(max_iter):
            clusters = [list() for i in range(k)]
            # Update cluster assignments
            for i in range(n):
                vec = feat_vecs[i]
                dist =  euclidean_distance(vec, centers, 1)
                min_index = np.argmin(dist, axis=0)
                clusters[min_index].append(i)
            # Restart algorithm in case of empty clusters
            if [] in clusters:
                print("WARNING: Empty cluster restarting...")
                proceed = True
                break
            # Update centers
            for i in range(k):
                indices = clusters[i]
                centers[i] = feat_vecs[indices].mean(axis=0)
    print("Final centers:\n", centers)
    return centers

def main():
    args = arg_handler()
    if args:
        all_descrs = get_all_descriptors(args.readfolder)

if __name__ == "__main__":
    main()