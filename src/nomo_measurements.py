import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.cluster import KMeans

from load import MeshMeasurements


if __name__ == '__main__':
    DIR = '/media/kristijan/kristijan-hdd-ex/datasets/NOMO/bodymeasurement_mat/male/'
    all_mvalues = []

    for fidx, fname in enumerate(os.listdir(DIR)):
        fpath = os.path.join(DIR, fname)
        mfile = loadmat(fpath)
        mnames = mfile['s'].dtype.names
        mvalues = []

        for midx in range(len(mnames)):
            mvalues.append(mfile['s'][0][0][midx][0][0])

        mvalues = np.array(mvalues)
        all_mvalues.append(mvalues)

    all_mvalues = np.nan_to_num(np.array(all_mvalues))
    
    kmeans = KMeans(n_clusters=194, random_state=0).fit(all_mvalues)
    labels = kmeans.labels_
    
    merged = np.concatenate([labels.reshape(-1, 1), all_mvalues], axis=1)
    df = pd.DataFrame(merged)
    row_names = ['cluster'] + list(mnames)
    df.to_excel('kmeans_nomo.xlsx', index=True, header=row_names)
