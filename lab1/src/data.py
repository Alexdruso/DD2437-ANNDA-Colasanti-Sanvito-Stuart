from typing import Tuple

import pandas as pd
import numpy as np


def generate_binary_classification_data(
        n: int = 200,
        class_ratio: float = 0.5,
        mA: Tuple[float] = (2.0, 0.5),
        mB: Tuple[float] = (-2.0, 0.0),
        sigmaA: float = 0.5,
        sigmaB: float = 0.5,
        classes: Tuple = (1, 0),
        reset_index: bool = False
) -> pd.DataFrame:
    size_positive_class = int(n * class_ratio)
    size_negative_class = n - size_positive_class

    result = pd.concat(
        [
            pd.DataFrame(
                {
                    'x1': np.random.normal(size=size_positive_class, loc=mA[0], scale=sigmaA),
                    'x2': np.random.normal(size=size_positive_class, loc=mA[1], scale=sigmaA),
                    'y': np.full(shape=size_positive_class, fill_value=classes[0])
                }
            ),
            pd.DataFrame(
                {
                    'x1': np.random.normal(size=size_negative_class, loc=mB[0], scale=sigmaB),
                    'x2': np.random.normal(size=size_negative_class, loc=mB[1], scale=sigmaB),
                    'y': np.full(shape=size_negative_class, fill_value=classes[1])
                }
            )
        ],
        axis=0
    )

    result = result.sample(frac=1)

    return result.reset_index() if reset_index else result
