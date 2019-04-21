# Standard library imports
import argparse

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from PIL import Image

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

# Read an image and return numpy array for color and grayscale
def read_image(path):
    image = Image.open(path)
    color=np.array(image.convert(mode='RGB'))
    gray=np.array(image.convert(mode='L'))
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