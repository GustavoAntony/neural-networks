from src.utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import dataframe_image as dfi
from sklearn.preprocessing import MinMaxScaler

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
    plt.close()

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
    plt.close()

def exercise3():
    df = pd.read_csv(os.path.join(DATA_OUTPUTS_FILE_PATH,'data','spaceship_titanic','train.csv'))
    df_head = df.head(5)
    dfi.export(df_head, os.path.join(IMAGES_OUTPUTS_FILE_PATH,'data','exercise3_1.png'))

    numerical_features = ['Age', 'RoomService','FoodCourt','ShoppingMall', 'Spa', 'VRDeck']
    categorical_features = ['HomePlanet','CryoSleep','Destination','VIP']
    special_columns = ['Name', 'Cabin', 'PassengerId']
    target_column = 'Transported'

    df.info()
    print(df.isnull().sum())

    df.drop(columns=['Name', 'PassengerId'], inplace=True)
    df[['Cabin_Deck', 'Cabin_Side']] = df['Cabin'].str.extract(r'([A-Z]+)\/\d+\/([A-Z]+)')
    df.drop(columns=['Cabin'], inplace=True)

    categorical_features.append('Cabin_Side')
    categorical_features.append('Cabin_Deck')

    for col in numerical_features:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])

    df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_features]), columns=numerical_features)

    df_processed = pd.concat([df_scaled, df_encoded], axis=1)
    df_processed[target_column] = df[target_column].astype(int) 
    
    df_processed_head = df_processed.head(5)
    dfi.export(df_processed_head, os.path.join(IMAGES_OUTPUTS_FILE_PATH,'data','exercise3_2.png'))

    fig, axes = plt.subplots(2, len(numerical_features), figsize=(20, 8))
    for i, col in enumerate(numerical_features):
        df[col].plot.hist(bins=30, ax=axes[0, i], alpha=0.7, color='skyblue')
        axes[0, i].set_title(f"{col} (Before Scaling)")
        df_scaled[col].plot.hist(bins=30, ax=axes[1, i], alpha=0.7, color='salmon')
        axes[1, i].set_title(f"{col} (After Scaling)")

    plt.tight_layout()
    plt.suptitle("Histograms of Numerical Features (Before and After Scaling)", y=1.02)
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'data','exercise3_4.png'))
    plt.close()


def main():
    exercise1()
    exercise2()
    exercise3()

if __name__ == "__main__":
    np.random.seed(42)
    main()