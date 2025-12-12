from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class RegressionTree:
    def __init__(self, X, Y):
        self.model = DecisionTreeRegressor()
        self.X = X
        self.Y = Y

    def fit(self):
        self.model.fit(self.X, self.Y)

    def predict(self, test_X):
        return self.model.predict(test_X)

    @staticmethod
    def metrics(test_y, pred_y):
        return mean_squared_error(test_y, pred_y)


