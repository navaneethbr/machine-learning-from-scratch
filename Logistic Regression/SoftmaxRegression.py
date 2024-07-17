import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
from sklearn.datasets import load_iris

class SoftmaxRegression:
    def __init__(self, epochs: int=1000, learning_rate: float=0.1) -> None:
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
        Y_onehot = onehot_encoder.fit_transform(y.reshape(-1,1))
        n = X.shape[0]
        # X = np.insert(X, 0, 1, axis=1)
        self.W = np.zeros((X.shape[1], Y_onehot.shape[1]))
        
        for _ in range(self.epochs):
            y_pred = softmax(X@self.W, axis=1)
            dw = (1/n)*(X.T@(y_pred - Y_onehot))
            step_w = self.lr*dw 
            self.W -= step_w 
            
        return -self.W
    
    def loss(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        y = onehot_encoder.fit_transform(y.reshape(-1,1))
        # X = np.insert(X, 0, 1, axis=1)
        z = - X @ self.W
        n = X.shape[0]
        loss = 1/n * (np.trace(X @ self.W @ y.T) + np.sum(np.log(np.sum(np.exp(z), axis=1))))
        return loss
            
if __name__ == "__main__":
    X = load_iris().data
    y = load_iris().target
    model = SoftmaxRegression()
    w = model.fit(X, y)
    print(w)