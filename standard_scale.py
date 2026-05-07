from sklearn import preprocessing
import torch

def scale(X):
    Scale_X = preprocessing.StandardScaler().fit(X)
    X_scale = Scale_X.transform(X)
    return X_scale

def soft_sh(w, t, device):
    return torch.sign(w)*torch.max((torch.abs(w) - t), torch.zeros(w.shape).to(device))

if __name__=='__main__':
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(5):
        x = i+1
        y = x+1