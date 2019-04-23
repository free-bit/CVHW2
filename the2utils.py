# Standard library imports
import argparse

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from PIL import Image

# Constants
PIPE_MODES = {
    "extract": False,
    "query": False
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
    parser.add_argument("--save",  help="Save all outputs", 
                       default=False, action="store_true")
    parser.add_argument("--show",  help="Show images found", 
                       default=False, action="store_true")
                       
    group = parser.add_argument_group(title='required arguments')
    group.add_argument("-rf", "--readfolder",  help="Directory of input files", 
                       metavar="FOLDER", type=check_path, required=True)
    group.add_argument("-c", "--filecount",  help="Number of input files to be sampled", 
                       metavar="COUNT", type=check_positive, required=True)
    group.add_argument("-p", "--pipemode",  help="Specify pipeline execution mode: \
                       (extract|query|all)", metavar="MODE", type=str, required=True)
    args = parser.parse_args()

    if args.help:
        parser.print_help()
    
    # Update pipeline flags
    args.pipemode = args.pipemode.lower()
    if (args.pipemode == "all"):
        PIPE_MODES["extract"] = True
        PIPE_MODES["query"] = True
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