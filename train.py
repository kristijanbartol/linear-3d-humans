import argparse
from sklearn.model_selection import train_test_split

from load import load
from metrics import evaluate
from models import Models
from logs import log
from visualize import visualize


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

    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    print(f'Preparing {args.dataset_name} dataset...')
    X, y, measurements, genders = load(args)
    print('Train/test splitting...')
    X_train, X_test, y_train, y_test, meas_train, meas_test, _, gender_test = train_test_split(
        X, y, measurements, genders, test_size=0.33, random_state=42)

    print(f'Creating {args.model} model...')
    model = getattr(Models, args.model)()
    print(f'Fitting model using {args.pose_reg_type}+{args.silh_reg_type}+{args.soft_reg_type} regressor...')
    print(f'Target variable: {args.target}...')
    reg = model.fit(X_train, y_train if args.target == 'shape' else meas_train)
    print('Predicting...')
    y_predict = reg.predict(X_test)
    print('Evaluating...')
    y_gt = y_test if args.target == 'shape' else meas_test
    params_errors, measurement_errors, s2s_dists = evaluate(y_predict, y_gt, gender_test)
    print('Logging to stdout...')
    log(model, args, params_errors, measurement_errors, s2s_dists)
    print('Visualizing...')
    visualize(model, args, params_errors, measurement_errors, s2s_dists)
