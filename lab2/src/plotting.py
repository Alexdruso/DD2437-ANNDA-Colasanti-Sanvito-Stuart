from typing import List
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

import pandas as pd


def plot_2d_function(
        dfs: List[pd.DataFrame],
        function_names: List[str],
        x: str = 'x',
        y: str = 'y',
        models: List = None,
        names: List = None,
        path: str = None
) -> None:
    if models is None: models = []
    if names is None: names = []

    plt.figure(figsize=(12, 7))

    # Plot the functions to be approximated

    for index in range(len(dfs)):
        df = dfs[index]
        plt.plot(df[x], df[y], label=function_names[index])

    # Plot the models' decision boundaries (if any)

    for index in range(len(models)):
        model = models[index]
        name = names[index]

        plt.plot(dfs[0][x], model.predict(dfs[0][[x]].to_numpy()), label=name, dashes=[6, 2])

    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()
    plt.legend()

    if path is not None:
        plt.savefig(path)

    plt.show()
