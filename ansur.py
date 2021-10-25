from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression 
import pandas as pd
import numpy as np


# TODO: Also use gender ----> all males.
# TODO: Also try using age.
input_keys = [
    'stature',
    'weightkg',
    'Age'
]


output_keys = [
    'acromialheight'
]


ansur = pd.read_csv('data/ansur.csv')

output_keys = list(ansur.keys()[1:93])

for out_key in output_keys:

    data_in = np.array([np.array(ansur[x], dtype=np.float32) for x in input_keys]).swapaxes(0, 1)
    #data_out = np.array([np.array(ansur[x], dtype=np.float32) for x in output_keys]).swapaxes(0, 1)
    data_out = np.array([np.array(ansur[x], dtype=np.float32) for x in [out_key]]).swapaxes(0, 1)

    x_train, x_test, y_train, y_test = train_test_split(data_in, data_out, test_size = 0.2, random_state=2021)

    regr = LinearRegression().fit(x_train, y_train)
    y_predict = regr.predict(x_test)

    means = np.mean(np.abs(y_predict - y_test), axis=0)
    stds = np.std(np.abs(y_predict - y_test), axis=0)

    #for idx, out_key in enumerate(output_keys):
    #    print(f'{out_key}: {means[idx]:.2f}, {stds[idx]:.2f}')

    score = r2_score(y_test, y_predict)
    print(f'R2-score ({out_key}): {score}:.4f (mean: {means}, std: {stds}')
