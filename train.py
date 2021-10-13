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
        '--model', type=str, choices=['linear', 'poly', 'tree', 'mlp'],
        help='machine learning model type'
    )

    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()

    print(f'Preparing {args.dataset_name} dataset...')
    X, y, measurements, genders = prepare(args)
    print('Train/test splitting...')
    X_train, X_test, y_train, y_test, _, gt_meas_test, _, gender_test = train_test_split(
        X, y, measurements, genders, test_size=0.33, random_state=42)

    print(f'Creating {args.model} model...')
    model = getattr(Models, args.model)()
    print(f'Fitting model using {args.pose_reg_type}+{args.silh_reg_type}+{args.soft_reg_type} regressor...')
    reg = model.fit(X_train, y_train)
    print('Predicting...')
    y_predict = reg.predict(X_test)
    print('Evaluating...')
    params_errors, measurement_errors, s2s_dists = evaluate(y_predict, y_test, gt_meas_test, gender_test)

    log(model, args, params_errors, measurement_errors, s2s_dists)
    
    visualize(model, args, params_errors, measurement_errors, s2s_dists)
