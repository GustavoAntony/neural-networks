from src.utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def exercise1():
    params = {
        0: {MEAN: [2, 3], STD: [0.8,2.5]},
        1: {MEAN: [5, 6], STD: [1.2,1.9]},
        2: {MEAN: [8, 1], STD: [0.9,0.9]},
        3: {MEAN: [15, 4], STD: [0.5,2]},
        }
    
    X = []
    y = []

    for cls, p in params.items():
        points = np.random.normal(loc=p["mean"], scale=p["std"], size=(100, 2))
        X.append(points)
        y.append(np.full(100, cls))

    X = np.vstack(X)
    y = np.hstack(y)

    colors = ['red', 'blue', 'green', 'purple']

    plt.figure(figsize=(8, 6))
    for cls in params.keys():
        plt.scatter(X[y == cls, 0], X[y == cls, 1], c=colors[cls], label=f"Class {cls}", alpha=0.6)

    plt.xlabel("Label 1")
    plt.ylabel("Label 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'data','exercise1_2.png'))

    plt.plot([2.5, 11.5], [-3, 9], color='black', linestyle='--', linewidth=2, label="Boundary 0-1")
    plt.plot([2.0, 4.5], [12.5, -0.2], color='black', linestyle='--', linewidth=2, label="Boundary 1-2")
    plt.plot([11.5, 11.5], [-2, 14], color='black', linestyle='--', linewidth=2, label="Boundary 2-3")
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'data','exercise1_3.png'))

def exercise2():
    params = {
        'A' : {MEAN : [0,0,0,0,0], COV : [[1,0.8,0.1,0,0],[0.8,1,0.3,0,0],[0.1,0.3,1,0.5,0],[0,0,0.5,1,0.2],[0,0,0,0.2,1.0]]},
        'B' : {MEAN : [1.5,1.5,1.5,1.5,1.5], COV : [[1.5,-0.7,0.2,0,0],[-0.7,1.5,0.4,0,0],[0.2,0.4,1.5,0.6,0],[0,0,0.6,1.5,0.3],[0,0,0,0.3,1.5]]}
    }

    cls_A = np.random.multivariate_normal(params['A'][MEAN], params['A'][COV], 500)
    cls_B = np.random.multivariate_normal(params['B'][MEAN], params['B'][COV], 500)

    labels_A = np.zeros(500)
    labels_B = np.ones(500)

    X = np.vstack((cls_A, cls_B))
    y = np.hstack((labels_A, labels_B))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Classe A', alpha=0.7)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Classe B', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'data','exercise2.png'))

def exercise3():
    pass

def main():
    exercise1()
    exercise2()
    exercise3()

if __name__ == "__main__":
    np.random.seed(42)
    main()