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

    def fit_normal_equation(self, X, y, is_polynomial=False, degree=2, reg_type='None', alpha=0.1):
        """
        Solves using Normal Equation (Inverse Matrix).
        Supports OLS and Ridge (L2). Lasso (L1) requires iterative solvers, so it's not here.
        """
        X_transformed = self._transform_features(X, is_polynomial, degree)
        X_b = self._add_intercept(X_transformed)
        y = y.flatten()

        X_T = X_b.T
        XtX = np.dot(X_T, X_b)
        
        # --- RIDGE REGRESSION (L2) ---
        # Formula: w = (X^T*X + alpha*I)^-1 * X^T*y
        if reg_type == 'Ridge (L2)':
            I = np.eye(XtX.shape[0])
            I[0, 0] = 0  # Do not regularize the bias term (intercept)
            XtX += alpha * I
        
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
            
        XtY = np.dot(X_T, y)
        self.weights = np.dot(XtX_inv, XtY)
        self.loss_history = []
        
        return {"Weights": self.weights}

    def fit_gd_stream(self, X, y, learning_rate=0.01, epochs=100, gd_type='batch', batch_size=32, is_polynomial=False, degree=2, reg_type='None', alpha=0.1):
        """
        Solves using Gradient Descent.
        Supports OLS, Ridge (L2), and Lasso (L1).
        """
        X_transformed = self._transform_features(X, is_polynomial, degree)
        y = y.flatten()
        X_b = self._add_intercept(X_transformed)
        m, n = X_b.shape
        
        self.weights = np.zeros(n)
        self.loss_history = []
        gd_type = gd_type.lower()

        for i in range(epochs):
            # 1. Data Selection
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

            # 2. Prediction & Error
            y_pred = np.dot(X_i, self.weights)
            error = y_pred - y_i
            
            # 3. Base Gradient (MSE Derivative)
            gradients = (1 / X_i.shape[0]) * np.dot(X_i.T, error)
            
            # 4. Add Regularization Term to Gradient
            if reg_type == 'Ridge (L2)':
                # Gradient of L2: alpha * w
                reg_term = (alpha / m) * self.weights
                reg_term[0] = 0 # Don't penalize bias
                gradients += reg_term
                
            elif reg_type == 'Lasso (L1)':
                # Gradient of L1: alpha * sign(w)
                reg_term = (alpha / m) * np.sign(self.weights)
                reg_term[0] = 0 # Don't penalize bias
                gradients += reg_term

            # 5. Update Weights
            self.weights -= learning_rate * gradients

            # 6. Calculate Total Loss (MSE + Penalty) for Visualization
            full_pred = np.dot(X_b, self.weights)
            mse_loss = np.mean((full_pred - y) ** 2)
            
            total_loss = mse_loss
            if reg_type == 'Ridge (L2)':
                total_loss += (alpha / (2*m)) * np.sum(np.square(self.weights[1:]))
            elif reg_type == 'Lasso (L1)':
                total_loss += (alpha / m) * np.sum(np.abs(self.weights[1:]))
                
            self.loss_history.append(total_loss)
            
            yield i, total_loss, self.loss_history, self.weights


class MyLogisticRegression:
    def __init__(self):
        self.weights = None
        self.loss_history = []
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
        
    def predict_proba(self, X):
        if self.weights is None:
            return None
        X_b = self._add_intercept(X)
        return self._sigmoid(np.dot(X_b, self.weights))
        
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit_perceptron_trick(self, X, y, learning_rate=0.1, epochs=1000):
        X_b = self._add_intercept(X)
        y = y.flatten()
        m, n = X_b.shape
        self.weights = np.zeros(n)
        self.loss_history = []
        
        for i in range(epochs):
            idx = np.random.randint(0, m)
            x_sample = X_b[idx]
            y_sample = y[idx]
            
            y_hat = 1 if np.dot(self.weights, x_sample) >= 0 else 0
            
            if y_hat != y_sample:
                if y_sample == 1:
                    self.weights += learning_rate * x_sample
                else:
                    self.weights -= learning_rate * x_sample
            
            full_preds = (np.dot(X_b, self.weights) >= 0).astype(int)
            loss = np.mean(y != full_preds)
            self.loss_history.append(loss)
            
            yield i, loss, self.loss_history, self.weights

    def fit_sigmoid_perceptron(self, X, y, learning_rate=0.1, epochs=1000):
        X_b = self._add_intercept(X)
        y = y.flatten()
        m, n = X_b.shape
        self.weights = np.zeros(n)
        self.loss_history = []
        
        for i in range(epochs):
            idx = np.random.randint(0, m)
            x_sample = X_b[idx]
            y_sample = y[idx]
            
            z = np.dot(x_sample, self.weights)
            y_hat = self._sigmoid(z)
            
            loss_grad = (y_hat - y_sample) * x_sample
            self.weights -= learning_rate * loss_grad
            
            epsilon = 1e-15
            full_z = np.dot(X_b, self.weights)
            full_y_hat = self._sigmoid(full_z)
            loss = -np.mean(y * np.log(full_y_hat + epsilon) + (1 - y) * np.log(1 - full_y_hat + epsilon))
            self.loss_history.append(loss)
            
            yield i, loss, self.loss_history, self.weights
            
    def fit_batch_logistic(self, X, y, learning_rate=0.1, epochs=1000):
        X_b = self._add_intercept(X)
        y = y.flatten()
        m, n = X_b.shape
        self.weights = np.zeros(n)
        self.loss_history = []
        
        for i in range(epochs):
            z = np.dot(X_b, self.weights)
            y_hat = self._sigmoid(z)
            
            gradient = np.dot(X_b.T, (y_hat - y)) / m
            self.weights -= learning_rate * gradient
            
            epsilon = 1e-15
            loss = -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
            self.loss_history.append(loss)
            
            yield i, loss, self.loss_history, self.weights


class DatasetManager:
    def load_sklearn_dataset(self, name):
        if name == "Diabetes (Reg)":
            data = datasets.load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df
        elif name == "California Housing (Reg)":
            data = datasets.fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df = df.sample(n=500, random_state=42) 
            df['target'] = data.target[:500]
            return df
        elif name == "Iris (Binary Class)":
            data = datasets.load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df = df[df['target'] != 2] 
            return df
        elif name == "Breast Cancer (Class)":
            data = datasets.load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df
        return None

    def create_synthetic_data(self, n_samples=300, noise=10, type='regression', n_features=1, n_informative=1):
        if type == 'regression':
            X, y = datasets.make_regression(
                n_samples=n_samples, 
                n_features=n_features, 
                n_informative=n_informative,
                noise=noise, 
                random_state=42
            )
            cols = [f"Feature_{i+1}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=cols)
            df['target'] = y
            return df
            
        elif type == 'classification':
            separation = max(0.5, 2.5 - (noise / 5.0))
            n_info = min(n_features, n_informative)
            n_redundant = 0
            
            X, y = datasets.make_classification(
                n_samples=n_samples, 
                n_features=n_features, 
                n_redundant=n_redundant, 
                n_informative=n_info, 
                random_state=42, 
                n_clusters_per_class=1, 
                class_sep=separation
            )
            cols = [f"Feature_{i+1}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=cols)
            df['target'] = y
            return df
            
        return None