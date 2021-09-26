import numpy as np
from scipy.signal import square
import pandas as pd

output_functions = {
    'sin': np.sin,
    'square': square
}


def generate_periodic_function(
        function: str = 'sin',
        custom_function=None,
        start: float = 0,
        stop: float = 2 * np.pi,
        step: float = 0.1,
        noise: bool = False,
        reset_index: bool = True
) -> pd.DataFrame:
    x = np.arange(start=start, stop=stop, step=step)

    output_function = output_functions[function] if function in output_functions else custom_function
    y = output_function(2 * x)

    df = pd.DataFrame(
                {
                    'x': x,
                    'y': y + np.random.normal(0, 0.1, len(y)) if noise else y
                }
            )

    return df.reset_index(drop=True) if reset_index else df
