import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 


input_keys = [
    'stature',
    'weightkg'
]


output_keys = [
    'headcircumference',
    'neckcircumference',
    'shoulder_to_crotch',
    'chestcircumference',
    'waistcircumference',
    'buttockcircumference',
    'wristcircumference',
    'bicepscircumferenceflexed',
    'forearmcircumferenceflexed',
    'arm_length',
    'inside_leg_length',
    'thighcircumference',
    'calfcircumference',
    'anklecircumference',
    'biacromialbreadth'
]


plot_names = [
    'Height',
    'Weight',
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'J',
    'K',
    'L',
    'M',
    'N',
    'O'
]



def visualize_measurement_distribution(gender, data_in, data_out):
    fig_name = f'ansur_distributions_{gender}.png'
    fig_path = os.path.join('vis/', fig_name)

    #labels = input_keys + output_keys
    data = np.concatenate([data_in, data_out], axis=1).swapaxes(0, 1)
    data[2:] /= 10.
    data[0, :] *= 100.
    #data[1, :] /= 100.
    measure_dict = dict(zip(
        plot_names, 
        data)
    )
    df = pd.DataFrame(measure_dict)
    df.boxplot(grid=False, figsize=(12, 5), rot=0)

    plt.title('ANSUR Males')
    plt.xlabel('Measurement Label')
    plt.ylabel('cm / kg')
    
    plt.savefig(fig_path)


if __name__ == '__main__':
    for gender in ['male', 'female']:
        print(f'{gender}\n==========')
        ansur = pd.read_csv(f'data/ansur-{gender}.csv', encoding='latin-1')
        
        ansur['arm_length'] = ansur['acromialheight'] - ansur['wristheight'] + ansur['handlength']
        ansur['inside_leg_length'] = ansur['crotchheight'] - ansur['lateralmalleolusheight']
        ansur['shoulder_to_crotch'] = ansur['sittingheight'] - (ansur['stature'] - ansur['acromialheight'])
        
        ansur['weightkg'] *= 0.1
        ansur['stature'] *= 0.001

        data_in = np.array([np.array(ansur[x], dtype=np.float32) for x in input_keys]).swapaxes(0, 1)
        data_out = np.array([np.array(ansur[x], dtype=np.float32) for x in output_keys]).swapaxes(0, 1)

        x_train, x_test, y_train, y_test = train_test_split(data_in, data_out, test_size = 0.2, random_state=2021)

        regr = LinearRegression().fit(x_train, y_train)
        y_predict = regr.predict(x_test)

        means = np.mean(np.abs(y_predict - y_test), axis=0)
        stds = np.std(np.abs(y_predict - y_test), axis=0)

        for idx, out_key in enumerate(output_keys):
            print(f'{out_key}: {means[idx]:.2f}, {stds[idx]:.2f}')
            
        visualize_measurement_distribution(gender, data_in, data_out)
