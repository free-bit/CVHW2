# Standard library imports
import argparse
import re
import sys

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

# Local imports
from PIL import Image

# Constants & global variables
PIPE_MODE = {
    "train": False,
    "x_valid": False,
    "test": False
}

# Disable multiple occurences of the same flag
class UniqueStore(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        if getattr(namespace, self.dest, self.default) is not None:
            parser.error(option_string + " appears several times.")
        setattr(namespace, self.dest, values)

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
    # Optional flags: Descriptor-related
    parser.add_argument("-d", "--dense",  help="Use dsift (default: false)",
                       default=False, action="store_true")
    parser.add_argument("-cc", "--corners",  help="Number of keypoints/corners to be kept" + \
                        " after each extraction (default: all)", metavar="COUNT", 
                        type=check_positive, default=None)
    parser.add_argument("--stepsize",  help="The step size to be used in dense sift," + \
                        " ignored if dense is false (default: 1)", metavar="STEP", 
                        type=check_positive, default=1)
    parser.add_argument("--fast",  help="Use fast dsift, ignored if dense is false (default: false)",
                       default=False, action="store_true")
    parser.add_argument("--percent",  help="Progress in %% for feature extraction (default:10)", 
                       metavar="VALUE", default=10, type=check_percent)
    # Optional flags: Others
    parser.add_argument("--save",  help="Save all outputs", 
                       default=False, action="store_true")
    parser.add_argument("--cmshow",  help="Show confusion matrix", 
                       default=False, action="store_true")
    parser.add_argument("--predonly",  help="If there is no ground truth, only make prediction," + \
                        " do NOT calculate accuracy (default: false)", default=False, 
                        action="store_true")
    # parser.add_argument("--debug", help="Debug (disable all flag checks)",
    #                     default=False, action="store_true")
    # Logic for flags
    enable_exec = ("-h" not in sys.argv) and ("--help" not in sys.argv) \
                  and ("--debug" not in sys.argv)
    enable_full = enable_exec and ("full" in sys.argv)
    enable_train_only = enable_exec and ("train" in sys.argv)
    enable_test_only = enable_exec and ("test" in sys.argv)
    # Determine the exact mode (TODO: Test)
    enable_x_valid = (not enable_full) and (not enable_test_only)\
                      and enable_train_only and ("-nf" in sys.argv or "--fold" in sys.argv)
    enable_train = (enable_train_only or enable_full)
    enable_test = (not enable_x_valid) and (enable_test_only or enable_full)

    group = parser.add_argument_group(title='required arguments')

    # Execution mode for the pipeline
    group.add_argument("-p", "--pipemode",  help="Specify pipeline execution mode", type=str,
                       choices=['train', 'test', 'full'], required=enable_exec, action=UniqueStore)
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
    test.add_argument("--bovwfile",  help="Saved BoVW file (required if pipe mode is test)", 
                      metavar="FILE", required=enable_test_only)
    test.add_argument("--vocabfile",  help="Saved vocabulary file (required if pipe mode is test)", 
                      metavar="FILE", required=enable_test_only)
    test.add_argument("-nk", "--knn",  help="k value for kNN", 
                      metavar="COUNT", type=check_positive, required=enable_test or enable_x_valid)
    args = parser.parse_args()
    # Print help if -h is used
    if args.help:
        parser.print_help()
        return

    # In case of debugging ignore all flags
    # if args.debug:
        # return

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

def save_pred_labels(path_to_save, names, top_voted):
    with open(path_to_save, "w+") as file:
        for i in range(len(names)):
            tmp = "{}: {}\n".format(names[i], top_voted[i])
            file.write(tmp)

# TESTED
def save_bovws(path_to_save, file_names, bovws):
    print("Saving BoVWs to the file: {}".format(path_to_save))
    with open(path_to_save, "w+") as file:
        for i in range(len(file_names)):
            np.savetxt(file, bovws[i], header=file_names[i])
    print("Saved.")

def save_vocab(path_to_save, vocab):
    print("Saving vocab to the file: {}".format(path_to_save))
    with open(path_to_save, "w+") as file:
        lines = "# k\n{}\n".format(vocab.n_clusters)
        file.write(lines)
        
        header = "center vectors"
        np.savetxt(file, vocab.cluster_centers_, header=header)
        
        header = "# inverted indices\n"
        file.write(header)
        for row in vocab.inv_indices_:
            np.savetxt(file, row, fmt="%d")
    print("Saved.")

def build_vocab_from_params(k, c_vecs, inv_indices):
    vocab = KMeans()
    vocab.n_clusters = k
    vocab.cluster_centers_ = c_vecs
    vocab.inv_indices_ = inv_indices
    return vocab

def read_vocab(path_to_read):
    """Reads a file containing vocabulary, returns vocab as KMeans object"""
    with open(path_to_read, "r") as file:
        # Discard comment
        file.readline()
        # Get k
        k = int(file.readline())

        # Discard comment
        file.readline()
        centers = np.empty((k, 128))
        # Get cluster centers
        for i in range(k):
            row = file.readline()
            centers[i] = np.fromstring(row, sep=' ')
        
        # Discard comment
        file.readline()
        # Get inverted indices
        inv_indices = [None] * k
        for i in range(k):
            row = file.readline()
            inv_indices[i] = np.fromstring(row, sep=' ')
        # Return KMeans object
        return build_vocab_from_params(k, centers, inv_indices)

# TESTED
def read_bovws(path_to_read):
    """Reads a file containing descriptors, returns name and matrix pairs as tuple"""
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

# Note that this function is for visualizing the confusion matrix and taken from: 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(c_mat, labels,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        c_mat = c_mat.astype('float') / c_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(c_mat)

    fig, ax = plt.subplots()
    im = ax.imshow(c_mat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # Show all ticks...
    ax.set(xticks=np.arange(c_mat.shape[1]),
           yticks=np.arange(c_mat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = c_mat.max() / 2.
    for i in range(c_mat.shape[0]):
        for j in range(c_mat.shape[1]):
            ax.text(j, i, format(c_mat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if c_mat[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

#UNUSED
def find_accuracy_old(test_labels, train_labels, pred_indices):
    """
    Get ground truth and test labels.
    Perform majority voting by using labels indicated by pred_indices to decide the label.
    In case of a tie get the minimum distanced label. (Counter class enables this behaviour.)
    Calculate and return accuracy.
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
    accuracy = accuracy_score(true_labels, top_voted)
    print("Score achieved: {}".format(accuracy))
    return accuracy