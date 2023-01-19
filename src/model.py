from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline


RANDOM_STATE = 37

MODELS = {
    'linear': LinearRegression(),
    'poly': make_pipeline(
        PolynomialFeatures(degree=4), 
        LinearRegression()
        ),
    'tree': DecisionTreeRegressor(random_state=RANDOM_STATE),
    'mlp': MLPRegressor(
        hidden_layer_sizes=(250), 
        random_state=RANDOM_STATE, max_iter=500
        )
}


class Model():
    
    def __init__(self, model_type='linear'):
        self.model_type = model_type
        self.model = MODELS[model_type]

    def feature_importances(self):
        if self.model_type == 'mlp':
            return None
        elif self.model_type == 'tree':
            return self.model.feature_importances_
        else:
            return self.model.coef_

    def intercepts(self):
        if self.model_type == 'mlp':
            return None
        elif self.model_type == 'tree':
            return None
        else:
            return self.model.intercept_
