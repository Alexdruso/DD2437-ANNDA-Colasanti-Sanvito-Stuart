import numpy as np
import matplotlib.pyplot as plt
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
    plt.scatter(negative[:, 0], negative[:, 1], label='Negative', marker='x', color='red')

    if positive_new is not None:
        plt.scatter(positive_new[:, 0], positive_new[:, 1], label='Unseen positive', color='blue', alpha=0.3)

    if negative_new is not None:
        plt.scatter(negative_new[:, 0], negative_new[:, 1], label='Unseen negative', marker='x', color='red', alpha=0.3)

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
