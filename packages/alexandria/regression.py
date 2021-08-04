import numpy as np
import pandas as pd

def stratify_continuous(bin_size: float, data, limit: float = None):

    categories = np.floor(data/bin_size)

    if limit:
        categories = np.where(categories>limit, limit, categories)
    
    if type(data) is pd.Series:
        categories = pd.Series(categories, name='cat_' + data.name, index=data.index)
    
    return categories