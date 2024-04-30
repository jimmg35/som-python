import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def export_cluster_result_to_csv(dataset, columns, path, clustered_name):
    columns = columns + ['cluster']
    # export clustered data
    df = pd.DataFrame(dataset, columns=columns).rename_axis("id")
    df.to_csv(os.path.join(path, clustered_name), encoding="utf-8")
