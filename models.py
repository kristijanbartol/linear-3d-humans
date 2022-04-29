from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline


class Models():
    
    RANDOM_STATE = 37

    @staticmethod
    def linear():
        return LinearRegression()

    @staticmethod
    def poly():
        return make_pipeline(PolynomialFeatures(degree=4), LinearRegression())

    @staticmethod
    def tree():
        return DecisionTreeRegressor(random_state=Models.RANDOM_STATE)

    @staticmethod
    def mlp():
        return MLPRegressor(hidden_layer_sizes=(250), random_state=Models.RANDOM_STATE, max_iter=500)

    @staticmethod
    def feature_importances(model):
        if type(model) == MLPRegressor:
            return None
        elif type(model) == DecisionTreeRegressor:
            return model.feature_importances_
        else:
            return model.coef_

    @staticmethod
    def intercepts(model):
        if type(model) == MLPRegressor:
            return None
        elif type(model) == DecisionTreeRegressor:
            return None
        else:
            return model.intercept_
