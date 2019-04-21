#!/usr/bin/env python3

# Standard library imports
import os

# Related third party imports
from cyvlfeat.sift import dsift, sift
from cyvlfeat.kmeans import kmeans

# Local imports
from the2utils import *

def kMeans():
    pass

def BoVW():
    pass

def applySIFT(image2D, **kwargs):
    dense = kwargs.get('dense', True)
    float = kwargs.get('float', False)
    frames = decrs = None
    if dense:
        fast = kwargs.get('fast', False)
        print(dense, float, fast)
        frames, decrs = dsift(image2D, fast=fast, float_descriptors=float)
    else:
        frames, decrs = sift(image2D, compute_descriptor=True, float_descriptors=float)
    return decrs

def kNN():
    arr = np.array([10,20,30,50,1,2])
    print(arr[np.argsort(arr)[-3:]])

def get_file_paths(folder):
    file_paths = []
    subfolders = os.listdir(folder)
    for subfolder in subfolders:
        subfolder = folder + subfolder + "/"
        file_paths.extend([(subfolder + file) for file in os.listdir(subfolder)])
    return file_paths

def main():
    args = arg_handler()
    if args:
        file_names = get_file_paths(args.readfolder)
        color, gray = read_image(file_names[0])

if __name__ == "__main__":
    main()
