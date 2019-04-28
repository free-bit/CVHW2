# Standard library imports
import argparse
import re
import sys

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from PIL import Image

# Constants & global variables
PIPE_MODE = {
    "train": False,
    "x_valid": False,
    "test": False
}

# Custom format for arg Help print
class CustomFormatter(argparse.HelpFormatter):
    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=100, # Modified
                 width=200):
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

def check_positive(value):
    try:
        value = int(value)
        assert (value > 0)
    except Exception as e:
        raise argparse.ArgumentTypeError("Positive integer is expected but got: {}".format(value))
    return value

def check_percent(value):
    value = check_positive(value)
    try:
        assert (value <= 100)
    except Exception as e:
        raise argparse.ArgumentTypeError("Value <= 100 but got: {}".format(value))
    return value

def check_path(path):
    if (path[-1] != "/"):
        path += "/"
    return path

# Handles cmd args
def arg_handler():
    parser = argparse.ArgumentParser(description='Bag of Visual Words', 
                                     formatter_class=CustomFormatter, 
                                     add_help=False)
    # Optional flags
    parser.add_argument("-h", "--help", help="Help message", action="store_true")
    parser.add_argument("--save",  help="Save all outputs", 
                       default=False, action="store_true")
    parser.add_argument("--show",  help="Show images found", 
                       default=False, action="store_true")
    # Descriptor-related
    parser.add_argument("-d", "--dense",  help="Use dsift (default: false)",
                       default=False, action="store_true")
    parser.add_argument("--fast",  help="Use fast dsift, ignored if dense is false (default: false)",
                       default=False, action="store_true")
    parser.add_argument("--percent",  help="Progress in %% for feature extraction (default:10)", 
                       metavar="VALUE", default=10, type=check_percent)
    # Debug mode (ignore all flags and run the script)
    parser.add_argument("--debug", help="Debug (disable all flag checks)",
                        default=False, action="store_true")
    # Logic for flags
    enable_exec = ("-h" not in sys.argv) and ("--help" not in sys.argv) \
                  and ("--debug" not in sys.argv)
    enable_full = ("full" in sys.argv)
    enable_train_only = ("train" in sys.argv)
    enable_test_only = ("test" in sys.argv)
    # Determine the exact mode (TODO: Test)
    enable_x_valid = enable_exec and (not enable_full) and (not enable_test_only)\
                     and enable_train_only and ("-nf" in sys.argv) or ("--fold" in sys.argv)
    enable_train = enable_exec and (enable_train_only or enable_full)
    enable_test = enable_exec and (not enable_x_valid) and (enable_test_only or enable_full)

    group = parser.add_argument_group(title='required arguments')

    # Execution mode for the pipeline
    group.add_argument("-p", "--pipemode",  help="Specify pipeline execution mode",
                       choices=['train', 'test', 'full'], required=enable_exec, type=str)
    # Train-specific
    train = parser.add_argument_group(title='train-specific')
    train.add_argument("--trainfolder",  help="Top level directory of training files", 
                        metavar="FOLDER", type=check_path, required=enable_train)
    train.add_argument("-nc", "--clusters",  help="Number of clusters (vocabulary size)", 
                       metavar="COUNT", type=check_positive, required=enable_train)
    train.add_argument("-nf", "--fold",  help="Number of folds for cross validation", 
                       metavar="COUNT", type=check_positive)
    train.add_argument("-cf", "--filecount",  help="Number of input files to be sampled", 
                       metavar="COUNT", type=check_positive, default=None)
    # Test-specific
    test = parser.add_argument_group(title='test-specific')
    test.add_argument("--testfolder",  help="Top level directory of test files", 
                       metavar="FOLDER", type=check_path, required=enable_test)
    test.add_argument("-nk", "--knn",  help="k value for kNN", 
                        metavar="COUNT", type=check_positive, required=enable_test)
    args = parser.parse_args()
    # Print help if -h is used
    if args.help:
        parser.print_help()
        return

    # In case of debugging ignore all flags
    if args.debug:
        return

    # Update pipeline flags accordingly
    PIPE_MODE["train"] = enable_train
    PIPE_MODE["x_valid"] = enable_x_valid
    PIPE_MODE["test"] = enable_test

    return args

# Read an image and return numpy array for color and grayscale
def read_image(path, **kwargs):
    read_color = kwargs.get('read_color', False)
    image = Image.open(path)
    color = None
    if(read_color):
        color = np.array(image.convert(mode='RGB'), dtype='Float32')
    gray = np.array(image.convert(mode='L'), dtype='Float32')
    return (color, gray)

# Show a grayscale image
def show_gray(image2D):
    if (image2D.ndim == 2):
        plt.imshow(image2D, cmap="gray")
        plt.show()
    else:
        print("2D grayscale image is expected")

# Show a color image
def show_color(image2D):
    if (image2D.ndim == 3):
        plt.imshow(image2D)
        plt.show()
    else:
        print("2D grayscale image is expected")

# Get euclidean distance between two feature vectors (n x m)
# axis = None: sum all values, return one value
# axis = 0   : sum values vertically, return m-many values
# axis = 1   : sum values horizontally, return n-many values
def euclidean_distance(vec1, vec2, axis=None):
    distance = np.sqrt(np.sum((vec1 - vec2)**2, axis=axis))
    return distance

# Normalize each row vector in the matrix
def l1_normalize(vec, axis=1):
    norm = np.linalg.norm(vec, ord=1, axis=axis)
    norm = norm.reshape((-1, 1))
    # Do not divide zero vectors
    return np.divide(vec, norm, where=norm!=0)

# TESTED
def save_bovws(path_to_save, file_names, bovws):
    print("Saving BoVWs to the file: {}".format(path_to_save))
    with open(path_to_save, "w+") as file:
        for i in range(len(file_names)):
            np.savetxt(file, bovws[i], header=file_names[i])
    print("Saved.")
# TESTED
def save_centers(path_to_save, centers):
    print("Saving cluster centers to the file: {}".format(path_to_save))
    with open(path_to_save, "w+") as file:
        header = "Cluster centers:"
        np.savetxt(file, centers, header=header)
    print("Saved.")
# TESTED
def save_inv_indices(path_to_save, inv_indices):
    print("Saving inverted indices to the file: {}".format(path_to_save))
    with open(path_to_save, "w+") as file:
        header = "# Inverted indices:\n"
        file.write(header)
        for cluster in inv_indices:
            np.savetxt(file, cluster, fmt="%d")
    print("Saved.")
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