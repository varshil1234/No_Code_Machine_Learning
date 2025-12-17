import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures

class MyLinearRegression:
    def __init__(self):
        self.weights = None
        self.loss_history = []
        self.poly = None

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _transform_features(self, X, is_polynomial=False, degree=2):
        if is_polynomial:
            self.poly = PolynomialFeatures(degree=degree, include_bias=False)
            return self.poly.fit_transform(X)
        return X

    def predict(self, X, is_polynomial=False):
        if self.weights is None:
            raise ValueError("Model is not fitted yet")
        
        if is_polynomial and self.poly:
            X = self.poly.transform(X)
            
        X_b = self._add_intercept(X)
        return np.dot(X_b, self.weights)

    def fit_normal_equation(self, X, y, is_polynomial=False, degree=2):
        X_transformed = self._transform_features(X, is_polynomial, degree)
        
        X_b = self._add_intercept(X_transformed)
        y = y.flatten()

        X_T = X_b.T
        XtX = np.dot(X_T, X_b)
        
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
            
        XtY = np.dot(X_T, y)
        self.weights = np.dot(XtX_inv, XtY)
        self.loss_history = []
        
        return {
            "X_augmented": X_b,
            "X_Transpose": X_T,
            "XtX": XtX,
            "XtX_Inverse": XtX_inv,
            "XtY": XtY,
            "Weights": self.weights
        }

    def fit_gd_stream(self, X, y, learning_rate=0.01, epochs=100, gd_type='batch', batch_size=32, is_polynomial=False, degree=2):
        X_transformed = self._transform_features(X, is_polynomial, degree)
        
        y = y.flatten()
        X_b = self._add_intercept(X_transformed)
        m, n = X_b.shape
        
        self.weights = np.zeros(n)
        self.loss_history = []
        gd_type = gd_type.lower()

        for i in range(epochs):
            if gd_type == 'batch':
                X_i, y_i = X_b, y
            elif 'stochastic' in gd_type:
                idx = np.random.randint(0, m)
                X_i, y_i = X_b[idx:idx+1], y[idx:idx+1] 
            elif 'mini' in gd_type:
                idx = np.random.choice(m, batch_size, replace=False)
                X_i, y_i = X_b[idx], y[idx]
            else:
                X_i, y_i = X_b, y

            y_pred = np.dot(X_i, self.weights)
            error = y_pred - y_i
            gradients = (1 / X_i.shape[0]) * np.dot(X_i.T, error)
            self.weights -= learning_rate * gradients

            full_pred = np.dot(X_b, self.weights)
            loss = np.mean((full_pred - y) ** 2)
            self.loss_history.append(loss)
            
            yield i, loss, self.loss_history, self.weights


class DatasetManager:
    def load_sklearn_dataset(self, name):
        if name == "Diabetes":
            data = datasets.load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df
        elif name == "California Housing":
            data = datasets.fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df = df.sample(n=500, random_state=42) 
            df['target'] = data.target[:500]
            return df
        return None

    def create_synthetic_data(self, n_samples=300, noise=10, is_curve=False):
        X = np.linspace(-3, 3, n_samples)
        
        if is_curve:
            y = 0.5 * X**2 + X + 2 + np.random.normal(0, noise, n_samples)
        else:
            y = 3 * X + 5 + np.random.normal(0, noise, n_samples)
            
        df = pd.DataFrame({'Feature_X': X, 'target': y})
        return df