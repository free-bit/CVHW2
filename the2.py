#!/usr/bin/env python3

# Standard library imports
import argparse
import os

# Related third party imports.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Custom format for arg Help print
class CustomFormatter(argparse.HelpFormatter):
    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=100, # Modified
                 width=None):
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)

            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append('%s' % option_string)
                parts[-1] += ' %s' % args_string
            return ', '.join(parts)

# Handles cmd args
def arg_handler():
    parser = argparse.ArgumentParser(description='Bag of Visual Words', 
                                     formatter_class=CustomFormatter, 
                                     add_help=False)
    parser.add_argument("-h", "--help", help="Help message", action="store_true")
    group = parser.add_argument_group(title='required arguments')
    group.add_argument("-rf", "--readfolder",  help="Directory of input files", metavar=("FOLDER"), type=str)
    args = parser.parse_args()
    
    # Checking args
    if args.help:
        parser.print_help()    
    if args.readfolder:
        return args

    return None

# dense=cv2.FeatureDetector_create("Dense")
# kp=dense.detect(imgGray)
# kp,des=sift.compute(imgGray,kp)

#Read an image and return numpy array for color and grayscale
def read_image(path):
    color = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (color, gray)

def kMeans():
    pass

def BoVW():
    pass

def kNN():
    pass

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

if __name__ == "__main__":
    main()

# Algorithm
