# Standard library imports
import argparse
import sys

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from PIL import Image

# Constants
PIPE_MODES = {
    "train": False,
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

def check_path(path):
    if (path[-1] != "/"):
        path += "/"
    return path

# Handles cmd args
def arg_handler():
    parser = argparse.ArgumentParser(description='Bag of Visual Words', 
                                     formatter_class=CustomFormatter, 
                                     add_help=False)
    parser.add_argument("-h", "--help", help="Help message", action="store_true")

    # Debug mode (ignore all flags and run the script)
    parser.add_argument("--debug", help="Debug (disable all flag checks)",
                        default=False, action="store_true")
    parser.add_argument("--save",  help="Save all outputs", 
                       default=False, action="store_true")
    parser.add_argument("--show",  help="Show images found", 
                       default=False, action="store_true")
    enable_pipe = ("-h" not in sys.argv) and ("--help" not in sys.argv) \
                  and ("--debug" not in sys.argv)
    group = parser.add_argument_group(title='required arguments')

    # Input execution mode for the pipeline
    group.add_argument("-p", "--pipemode",  help="Specify pipeline execution mode",
                       choices=['train', 'test', 'both'], required=enable_pipe, type=str)
    group.add_argument("-c", "--filecount",  help="Number of input files to be sampled", 
                       metavar="COUNT", type=check_positive, required=("train" in sys.argv))
    group.add_argument("--trainfolder",  help="Top level directory of training files", 
                       metavar="FOLDER", type=check_path, required=("train" in sys.argv))
    group.add_argument("--testfolder",  help="Top level directory of test files", 
                       metavar="FOLDER", type=check_path, required=("test" in sys.argv))
    args = parser.parse_args()
    # Print help if -h is used
    if args.help:
        parser.print_help()
        return

    # In case of debugging ignore all flags
    if args.debug:
        return

    # Update pipeline flags accordingly
    args.pipemode = args.pipemode.lower()
    if (args.pipemode == "all"):
        PIPE_MODES["train"] = True
        PIPE_MODES["test"] = True
    else:
        PIPE_MODES[args.pipemode] = True

    return args

# Read an image and return numpy array for color and grayscale
def read_image(path):
    image = Image.open(path)
    color = np.array(image.convert(mode='RGB'))
    gray = np.array(image.convert(mode='L'))
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