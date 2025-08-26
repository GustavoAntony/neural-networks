from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt

def exercise1():
    params = {
        0: {MEAN: [2, 2], STD: 0.8},
        1: {MEAN: [6, 2], STD: 0.8},
        2: {MEAN: [2, 6], STD: 0.8},
        3: {MEAN: [6, 6], STD: 0.8},
        }
    
    np.random.seed(42)
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

    plt.axvline(x=4, color='black', linestyle='--')
    plt.axhline(y=4, color='black', linestyle='--')

    plt.xlabel("Label 1")
    plt.ylabel("Label 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUTS_FILE_PATH,'exercise1.png'))

def main():
    exercise1()

if __name__ == "__main__":
    main()