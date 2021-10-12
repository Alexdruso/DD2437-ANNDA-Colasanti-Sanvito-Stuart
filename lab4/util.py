import matplotlib.pyplot as plt
import numpy as np


def sigmoid(support):
    """
    Sigmoid activation function that finds probabilities to turn ON each unit.

    Args:
      support: shape is (size of mini-batch, size of layer)
    Returns:
      on_probabilities: shape is (size of mini-batch, size of layer)
    """

    on_probabilities = 1. / (1. + np.exp(-support))
    return on_probabilities


def softmax(support):
    """
    Softmax activation function that finds probabilities of each category

    Args:
      support: shape is (size of mini-batch, number of categories)
    Returns:
      probabilities: shape is (size of mini-batch, number of categories)
    """

    expsup = np.exp(support - np.max(support, axis=1)[:, None])
    return expsup / np.sum(expsup, axis=1)[:, None]


def sample_binary(on_probabilities):
    """
    Sample activations ON=1 (OFF=0) from probabilities sigmoid probabilities

    Args:
      support: shape is (size of mini-batch, size of layer)
    Returns:
      activations: shape is (size of mini-batch, size of layer)
    """

    activations = 1. * (on_probabilities >=
                        np.random.random_sample(size=on_probabilities.shape))
    return activations


def sample_categorical(probabilities):
    """
    Sample one-hot activations from categorical probabilities

    Args:
      support: shape is (size of mini-batch, number of categories)
    Returns:
      activations: shape is (size of mini-batch, number of categories)
    """

    cumsum = np.cumsum(probabilities, axis=1)
    rand = np.random.random_sample(size=probabilities.shape[0])[:, None]
    activations = np.zeros(probabilities.shape)
    activations[range(probabilities.shape[0]),
                np.argmax((cumsum >= rand), axis=1)] = 1
    return activations


def load_idxfile(filename):
    """
    Load idx file format. For more information : http://yann.lecun.com/exdb/mnist/
    """
    import struct
    import os

    with open(os.path.join(os.getcwd(), 'lab4', filename), 'rb') as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0:
            raise Exception('Invalid idx file: unexpected magic number!')
        dtype, ndim = ord(_file.read(1)), ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0]
                 for _ in range(ndim)]
        data = np.fromfile(_file, dtype=np.dtype(
            np.uint8).newbyteorder('>')).reshape(shape)
    return data


def read_mnist(dim=None, n_train=60000, n_test=1000):
    """
    Read mnist train and test data. Images are normalized to be in range [0,1]. Labels are one-hot coded.
    """

    if dim is None:
        dim = [28, 28]
    train_imgs = load_idxfile("train-images-idx3-ubyte")
    train_imgs = train_imgs / 255.
    train_imgs = train_imgs.reshape(-1, dim[0] * dim[1])

    train_lbls = load_idxfile("train-labels-idx1-ubyte")
    train_lbls_1hot = np.zeros((len(train_lbls), 10), dtype=np.float32)
    train_lbls_1hot[range(len(train_lbls)), train_lbls] = 1.

    test_imgs = load_idxfile("t10k-images-idx3-ubyte")
    test_imgs = test_imgs / 255.
    test_imgs = test_imgs.reshape(-1, dim[0] * dim[1])

    test_lbls = load_idxfile("t10k-labels-idx1-ubyte")
    test_lbls_1hot = np.zeros((len(test_lbls), 10), dtype=np.float32)
    test_lbls_1hot[range(len(test_lbls)), test_lbls] = 1.

    return train_imgs[:n_train], train_lbls_1hot[:n_train], test_imgs[:n_test], test_lbls_1hot[:n_test]


def viz_rf(weights, it, grid):
    """
    Visualize receptive fields and save
    """
    fig, axs = plt.subplots(grid[0], grid[1], figsize=(
        grid[1], grid[0]))  # ,constrained_layout=True)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    imax = abs(weights).max()
    for x in range(grid[0]):
        for y in range(grid[1]):
            axs[x, y].set_xticks([])
            axs[x, y].set_yticks([])
            axs[x, y].imshow(weights[:, :, y + grid[1] * x],
                             cmap="bwr", vmin=-imax, vmax=imax, interpolation=None)
    plt.savefig("./lab4/images/rf.iter%06d.png" % it)
    plt.close('all')


def stitch_video(fig, imgs):
    """
    Stitches a list of images and returns a animation object
    """
    import matplotlib.animation as animation

    return animation.ArtistAnimation(fig, imgs, interval=100, blit=True, repeat=False)
