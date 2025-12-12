from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

class LinearModel:
    def __init__(self, X, Y):
        self.model = linear_model.LinearRegression()
        self.X = X
        self.Y = Y

    def fit(self):
        self.model.fit(self.X, self.Y)

    def predict(self, test_X):
        return self.model.predict(test_X)

    @staticmethod
    def metrics(test, pred):
        return (mean_squared_error(test, pred),
                r2_score(test, pred))

