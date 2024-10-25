import pandas as pd
import torch
import numpy as np

class GaussianNB():
    def __init__(self):
        self.X_positive_mean = 0
        self.X_negative_mean = 0
        self.X_positive_std = 0
        self.X_negative_std = 0
        self.X_positive_ratio = 1
        self.X_negative_ratio = 0

    def fit(self, X, y):
        X = torch.FloatTensor(X)
        X_positive = X[y==1]
        X_negative = X[y==-1]
        print("X_positive: ", X_positive)
        # print("shape: ", X_positive.shape)
        print('mean')
        print(torch.mean(X_positive, axis=0))
        self.X_positive_mean = torch.mean(X_positive, axis=0)
        self.X_negative_mean = torch.mean(X_negative, axis=0)
        self.X_positive_std = torch.std(X_positive, axis=0)
        self.X_negative_std = torch.std(X_negative, axis=0)
        x_positive_sample_size = X_positive.shape[0]
        self.X_positive_ratio = x_positive_sample_size/X.shape[0]
        self.X_negative_ratio = 1 - x_positive_sample_size/X.shape[0]

    def predict(self, X):
        X = torch.FloatTensor(X)
        positive = self._calculateProbability(X, self.X_positive_mean, self.X_positive_std) * self.X_positive_ratio
        negative = self._calculateProbability(X, self.X_negative_mean, self.X_negative_std) * self.X_negative_ratio
        print('positive')
        print(positive)
        print(torch.sigmoid(positive))
        positive = torch.sigmoid(positive)
        print('negative')
        print(negative)
        print(torch.sigmoid(negative))
        print(torch.prod(positive, 1))
        print(torch.prod(negative, 1))

                
    def _calculateProbability(self, x, mean, std):
        return np.exp(-(x - mean)**2 / (2 * std**2)) / (np.sqrt(2 * np.pi) * std)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# X = torch.tensor([[3, 2, 1, 0, 1], [1, 2, 1, 2, 1], [1, 0, 1, 2, 1], [2, 2, 0, 0, 1], [3, 1, 1, 0, 1]],dtype=torch.float16)
# y = torch.tensor([1, -1, -1, 1, 1])


# cls = GaussianNB()
# GaussianNB.fit(X, y)







































    #     data = pd.concat([X, y], axis=1)

    #     self.features_stats = data.groupby(y.name).agg(["mean", "std"])
    #     self.class_probs = data.groupby(y.name).size() / data.shape[0]
    #     self.unique_y_labels = self.class_probs.index.values
    
    # def predict_proba(self, X):
    #     class_probs = []
    #     for i in self.unique_y_labels:
    #         features_prob = X.apply(lambda x: self._calculateProbability(x, self.features_stats.loc[i, x.name]["mean"], self.features_stats.loc[i, x.name]["std"]))
    #         class_probs.append(np.product(features_prob, axis=1) * self.class_probs[i])
    #     class_probs = pd.concat(class_probs, axis=1)
    #     normalized_class_probs = class_probs.mul(1 / class_probs.sum(axis=1), axis=0)
    #     return normalized_class_probs
    
    # def predict(self, X):
    #     return np.argmax(self.predict_proba(X).values, axis=1)
            
        
    # def _calculateProbability(self, x, mean, std):
    #     return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-(x - mean)**2 / (2 * std**2))