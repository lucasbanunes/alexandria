from typing import List, Union


def set_seeds(seed: int, packages: Union[List[str], str]) -> None:
    """
    Sets the seed for the random number generators of the specified packages.
    This function is useful to ensure reproducibility of the experiments.
    Currently supports the following packages: 'numpy', 'random', 'tensorflow'.

    Parameters
    ----------
    seed : int
        Seed for the random number generators.
    packages : Union[List[str], str]
        Names of the packages to set the seed for.
        If 'all', sets the seed for all supported packages.

    Raises
    ------
    ValueError

    """
    if packages == 'all':
        packages = ['numpy', 'random', 'tensorflow']
    for package_name in packages:
        match package_name:
            case 'numpy':
                import numpy as np
                np.random.seed(seed)
            case 'random':
                import random
                random.seed(seed)
            case 'tensorflow':
                import tensorflow as tf
                tf.random.set_seed(seed)
            case _:
                raise ValueError(f'Package {package_name} is not supported.')
