from typing import Tuple

import pandas as pd
import numpy as np


def generate_binary_classification_data(
        n: int = 100,
        mA: Tuple[float] = (1.0, 0.5),
        mB: Tuple[float] = (-1.0, 0.0),
        sigmaA: float = 0.5,
        sigmaB: float = 0.5
) -> pd.DataFrame:
    result = pd.concat(
        [
            pd.DataFrame(
                {
                    'x1': np.random.normal(size=n) * sigmaA + mA[0],
                    'x2': np.random.normal(size=n) * sigmaA + mA[1],
                    'y': np.ones(shape=n)
                }
            ),
            pd.DataFrame(
                {
                    'x1': np.random.normal(size=n) * sigmaB + mB[0],
                    'x2': np.random.normal(size=n) * sigmaB + mB[1],
                    'y': np.zeros(shape=n)
                }
            )
        ],
        axis=0
    )

    return result.sample(frac=1)
