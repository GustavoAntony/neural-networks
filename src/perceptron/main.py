import numpy as np
from src.utils import *
import matplotlib.pyplot as plt
from src.perceptron.perceptron import Perceptron

PARAMS_EX1 = {
        0 : {
            MEAN : [1.5, 1.5], 
            STD : [[0.5, 0], [0, 0.5]]
        },
        1 : {
            MEAN : [5, 5], 
            STD : [[0.5, 0], [0, 0.5]]
        }
    }

PARAMS_EX2 = {
        0 : {
            MEAN : [3, 3], 
            STD : [[1.5, 0], [0, 1.5]]
        },
        1 : {
            MEAN : [4, 4], 
            STD : [[1.5, 0], [0, 1.5]]
        }
    }

def generate_data(params, n_samples = 1000):
    X = []
    y = []

    cls_A = np.random.multivariate_normal(params[0][MEAN], params[0][STD], n_samples)
    cls_B = np.random.multivariate_normal(params[1][MEAN], params[1][STD], n_samples)
    labels_A = np.zeros(n_samples)
    labels_B = np.ones(n_samples)

    X = np.vstack((cls_A, cls_B))
    y = np.hstack((labels_A, labels_B))

    return X, y

def save_plot_data(X, y, params, img_name):
    colors = ['red', 'blue']

    plt.figure(figsize=(8, 6))
    for cls in params.keys():
        plt.scatter(X[y == cls, 0], X[y == cls, 1], c=colors[cls], label=f"Class {cls}", alpha=0.6)

    plt.xlabel("Label 1")
    plt.ylabel("Label 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'perceptron',img_name))  

def exercise1():
    X, y  = generate_data(params=PARAMS_EX1, n_samples=1000)
    save_plot_data(X, y, PARAMS_EX1,'exercise1_1.png')
    perceptron = Perceptron('ex1')
    perceptron.train(X, y)

def exercise2():
    X2, y2 = generate_data(params=PARAMS_EX2, n_samples=1000)
    save_plot_data(X2, y2, PARAMS_EX2, 'exercise2_1.png')
    perceptron = Perceptron('ex2')
    perceptron.train(X2,y2)

def main():
    exercise1()
    exercise2()

if __name__ == "__main__":
    np.random.seed(42)
    main()

