import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from typing import List


def plot_decision_boundary(
        X: np.array,
        positive: np.array,
        negative: np.array,
        positive_new: np.array = None,
        negative_new: np.array = None,
        models: List = None,
        names: List = None,
        path: str = None
) -> None:
    if models is None:
        models = []
    if names is None:
        names = []
    plt.figure(figsize=(12, 7))
    plt.scatter(positive[:, 0], positive[:, 1], label='Positive', color='blue')
    plt.scatter(negative[:, 0], negative[:, 1],
                label='Negative', marker='x', color='red')

    if positive_new is not None:
        plt.scatter(positive_new[:, 0], positive_new[:, 1],
                    label='Unseen positive', color='blue', alpha=0.3)

    if negative_new is not None:
        plt.scatter(negative_new[:, 0], negative_new[:, 1],
                    label='Unseen negative', marker='x', color='red', alpha=0.3)

    # Plot the models' decision boundaries (if any)

    for index in range(len(models)):
        model = models[index]
        name = names[index]

        coef = model.coef_.flatten()  # weights
        w0 = model.intercept_  # bias
        w1 = coef[0]
        w2 = coef[1]

        step = 100
        ds_x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), step)
        # Compute x2 component given some x1:
        # w^T x + x0 = 0 -> w0 + w1 * x1 + w2 * x2 = 0 -> x2 = - (w0 + w1*x1) / w2
        ds_x2 = [-(w0 + w1 * x1) / w2 for x1 in ds_x1]
        plt.plot(ds_x1, ds_x2, label=name)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.legend()

    if path is not None:
        plt.savefig(path)

    plt.show()


def plot_learning_curve(
        errors: List,
        names: List,
        metric: str,
        path: str = None
) -> None:
    plt.figure(figsize=(12, 7))

    for index in range(len(errors)):
        error = errors[index]
        name = names[index]
        ds_x1 = [i for i in range(len(error))]
        ds_x2 = error
        plt.plot(ds_x1, ds_x2, label=name)

    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.grid()
    plt.legend()

    if path is not None:
        plt.savefig(path)

    plt.show()


def plot_decision_boundary_tlp(
        X: np.array,
        y: np.array,
        model,
        path: str = None
) -> None:

    fig, ax = plt.subplots(figsize=(12, 7))

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02  # step size
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, h),
                                   np.arange(x2_min, x2_max, h))
    X_mesh = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(X_mesh)

    # Set colormap
    cmap = cm.coolwarm_r
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0.3, 0.3, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    y_pred = y_pred.reshape(x1_mesh.shape)
    ax.contourf(x1_mesh, x2_mesh, y_pred, cmap=my_cmap)

    negative = X[(y == -1).flatten()]
    positive = X[(y == 1).flatten()]

    ax.scatter(positive[:, 0], positive[:, 1], label='Positive', color='blue')
    ax.scatter(negative[:, 0], negative[:, 1],
               label='Negative', marker='x', color='red')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(False)
    plt.legend()

    if path is not None:
        plt.savefig(path)

    plt.show()
