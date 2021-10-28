import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from load import load, load_star
from metrics import evaluate
from logs import log
from visualize import visualize


MODEL_PATH_TEMPLATE = './models/{}_{}_{}.sav'


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', type=str, 
        help='root data folder'
    )
    parser.add_argument(
        '--dataset_name', type=str, 
        help='dataset name'
    )
    parser.add_argument(
        '--data_type', type=str, choices=['gt', 'est'],
        help='whether to use ground truth regressor'
    )
    parser.add_argument(
        '--pose_reg_type', type=str,
        help='regressor type defined by the set of pose input features'
    )
    parser.add_argument(
        '--silh_reg_type', type=str,
        help='regressor type defined by the set of pose input features'
    )
    parser.add_argument(
        '--soft_reg_type', type=str,
        help='regressor type defined by the set of soft input features (weight etc.)'
    )
    parser.add_argument(
        '--target', type=str, choices=['shape', 'measurements'],
        help='target variable'
    )
    parser.add_argument(
        '--model', type=str, choices=['linear', 'poly', 'tree', 'mlp'],
        help='machine learning model type'
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'test'],
        help='train or test'
    )
    parser.add_argument(
        '--gender', type=str, choices=['male', 'female', 'neutral'],
        help='train or test'
    )

    return parser


def train(args):
    print(f'Preparing {args.dataset_name} dataset...')
    if args.dataset_name != 'star':
        X, y, measurements, genders = load(args)
    else:
        X, y, measurements, genders = load_star(args)
    print('Train/test splitting...')
    X_train, X_test, y_train, y_test, meas_train, meas_test, _, gender_test = train_test_split(
        X, y, measurements, genders, test_size=0.33, random_state=2021)

    print(f'Creating {args.model} model...')
    model = LinearRegression()
    print(f'Fitting model using {args.pose_reg_type}+{args.silh_reg_type}+{args.soft_reg_type} regressor...')
    print(f'Target variable: {args.target}...')
    reg = model.fit(X_train, y_train if args.target == 'shape' else meas_train)
    print('Saving model...')
    pickle.dump(model, open(MODEL_PATH_TEMPLATE.format(args.dataset_name, args.target, args.gender), 'wb'))
    print('Predicting...')
    y_predict = reg.predict(X_test)
    print('Evaluating...')
    y_gt = y_test if args.target == 'shape' else meas_test
    params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds = evaluate(y_predict, y_gt, gender_test, args.target)


    score = r2_score(y_gt, y_predict)
    print(f'R2-score: {score}')


    print('Logging to stdout...')
    log(model, args, params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds)
    print('Visualizing...')
    visualize(model, args, params_errors, measurement_errors, s2s_dists)


def test(args):
    print(f'Preparing {args.dataset_name} dataset...')
    X_test, y_test, meas_test, genders = load_star(args)
    print('Loading model...')
    model = pickle.load(open(MODEL_PATH_TEMPLATE.format('caesar', args.target, args.gender), 'rb'))
    print('Predicting...')
    y_predict = model.predict(X_test)

    print('Evaluating...')
    y_gt = y_test if args.target == 'shape' else meas_test
    params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds = evaluate(y_predict, y_gt, genders, args.target)

    print('Logging to stdout...')
    log(model, args, params_errors, params_stds, measurement_errors, measurement_stds, s2s_dists, s2s_stds)
    print('Visualizing...')
    visualize(model, args, params_errors, measurement_errors, s2s_dists)


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    locals()[args.mode](args)
