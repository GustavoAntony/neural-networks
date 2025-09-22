import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from src.mlp.mlp import MLP
from src.utils import *
import os
from sklearn.decomposition import PCA

def generate_binary_data():
    """Gera dados para o Exercício 2 (classificação binária)."""
    n_samples = 1000
    random_state = 42

    # Classe 0: 1 cluster
    X0, y0 = make_classification(n_samples=int(n_samples * 0.5), n_features=2, n_informative=2,
                                 n_redundant=0, n_clusters_per_class=1, flip_y=0.1,
                                 class_sep=0.8, random_state=random_state)
    y0[:] = 0

    # Classe 1: 2 clusters
    X1, y1 = make_classification(n_samples=int(n_samples * 0.5), n_features=2, n_informative=2,
                                 n_redundant=0, n_clusters_per_class=2, flip_y=0.1,
                                 class_sep=0.8, random_state=random_state + 1)
    y1[:] = 1

    # Combina e embaralha os dados
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    
    shuffle_indices = np.random.permutation(n_samples)
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    plt.scatter(X_train[:,0], X_train[:,1], c=y_train.flatten(), cmap='bwr', alpha=0.6)
    plt.title('Training Data - Exercise 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'mlp','exercise2_data.png'))
    return X_train, X_test, y_train, y_test

def generate_multiclass_data():
    """Gera dados para os Exercícios 3 e 4 (classificação multi-classe)."""
    n_samples = 1500
    n_features = 4
    random_state = 42

    # Classe 0: 2 clusters
    X0, y0 = make_classification(n_samples=int(n_samples/3), n_features=n_features, n_informative=n_features,
                                 n_redundant=0, n_clusters_per_class=2, flip_y=0.01,
                                 class_sep=1.0, random_state=random_state)
    y0[:] = 0

    # Classe 1: 3 clusters
    X1, y1 = make_classification(n_samples=int(n_samples/3), n_features=n_features, n_informative=n_features,
                                 n_redundant=0, n_clusters_per_class=3, flip_y=0.01,
                                 class_sep=1.0, random_state=random_state + 1)
    y1[:] = 1

    # Classe 2: 4 clusters
    X2, y2 = make_classification(n_samples=int(n_samples/3), n_features=n_features, n_informative=n_features,
                                 n_redundant=0, n_clusters_per_class=4, flip_y=0.01,
                                 class_sep=1.0, random_state=random_state + 2)
    y2[:] = 2

    # Combina e embaralha os dados
    X = np.vstack((X0, X1, X2))
    y = np.hstack((y0, y1, y2))
    
    shuffle_indices = np.random.permutation(n_samples)
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    # Converte y para one-hot encoding
    y_one_hot = np.eye(3)[y]
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=random_state)
    
    # Armazena os rótulos originais para avaliação de acurácia
    y_test_labels = np.argmax(y_test, axis=1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    plt.scatter(X_train_2d[:,0], X_train_2d[:,1], c=np.argmax(y_train, axis=1), cmap='viridis', alpha=0.6)
    plt.title('Training Data (PCA) - Exercise 3/4')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'mlp','exercise3_data_pca.png'))
    plt.close()
    return X_train, X_test, y_train, y_test, y_test_labels

def plot_decision_boundary(model, X, y, image_name):
    """
    Plota o limite de decisão para uma MLP treinada em dados 2D.
    Funciona para modelos binários e multi-classe.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    # Prepara os dados de entrada para o modelo
    Z_input = np.c_[xx.ravel(), yy.ravel()]
    Z_pred = model.forward(Z_input)
    
    # Converte a saída para rótulos de classe
    if model.task == 'binary':
        Z = (Z_pred > 0.5).astype(int).reshape(xx.shape)
        cmap = plt.cm.Spectral
        y_labels = y.flatten()
    else: # multiclasse
        Z = np.argmax(Z_pred, axis=1).reshape(xx.shape)
        cmap = plt.cm.get_cmap('viridis', model.output_size)
        y_labels = np.argmax(y, axis=1)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y_labels, s=40, cmap=cmap)
    plt.title("MLP Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    #save image

    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'mlp',image_name))
    plt.close()

def plot_decision_boundary_pca(model, X, y, pca, image_name):
    """
    Plota o limite de decisão para uma MLP treinada em dados multi-dimensionais,
    projetando os dados e a fronteira de decisão no espaço 2D do PCA.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Gera grid no espaço PCA
    X_pca = pca.transform(X)
    x1_min, x1_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    x2_min, x2_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
    grid_pca = np.c_[xx.ravel(), yy.ravel()]

    # Inverte a projeção PCA para o espaço original
    grid_original = pca.inverse_transform(grid_pca)
    Z_pred = model.forward(grid_original)
    Z = np.argmax(Z_pred, axis=1).reshape(xx.shape)
    y_labels = np.argmax(y, axis=1)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.get_cmap('viridis', model.output_size))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_labels, s=40, cmap=plt.cm.get_cmap('viridis', model.output_size))
    plt.title("MLP Decision Boundary (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'mlp',image_name))
    plt.close()

def exercise_2():
    print("--- Executando Exercício 2: Classificação Binária ---")
    X_train, X_test, y_train, y_test = generate_binary_data()

    input_size = X_train.shape[1]

    # hidden_sizes = [8]
    # output_size = 1
    # learning_rate = 0.1
    # epochs = 500

    hidden_sizes = [16]  # Increased from 8
    output_size = 1
    learning_rate = 0.05 # Adjusted from 0.1
    epochs = 2000        # Increased from 500

    mlp = MLP(input_size, hidden_sizes, output_size, learning_rate, task='binary')
    train_accuracies = []
    for epoch in range(epochs):
        mlp.train(X_train, y_train, 1)
        y_train_pred = mlp.forward(X_train)
        y_train_pred_classes = (y_train_pred > 0.5).astype(int)
        acc = accuracy_score(y_train, y_train_pred_classes)
        train_accuracies.append(acc)
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {acc*100:.2f}%")

    y_test_pred = mlp.forward(X_test)
    y_test_pred_classes = (y_test_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_test_pred_classes)
    print(f"\nAcurácia do Teste (Exercício 2): {accuracy * 100:.2f}%\n")

    # Plot training accuracy over epochs
    plt.figure(figsize=(8,4))
    plt.plot(train_accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'mlp','exercise2_accuracy.png'))
    plt.close()

    # Plota o limite de decisão se a dimensão for 2
    if input_size == 2:
        plot_decision_boundary(mlp, X_test, y_test, 'exercise2_decision_boundary.png')

def exercise_3():
    print("\n--- Executando Exercício 3: Classificação Multi-classe (1 camada) ---")
    X_train, X_test, y_train, y_test, y_test_labels = generate_multiclass_data()

    input_size = X_train.shape[1]

    # hidden_sizes = [16]
    # output_size = 3
    # learning_rate = 0.05
    # epochs = 500

    hidden_sizes = [64]  
    output_size = 3
    learning_rate = 0.05
    epochs = 2000    

    mlp = MLP(input_size, hidden_sizes, output_size, learning_rate, task='multiclass')
    train_accuracies = []
    for epoch in range(epochs):
        mlp.train(X_train, y_train, 1)
        y_train_pred = mlp.forward(X_train)
        y_train_pred_labels = np.argmax(y_train_pred, axis=1)
        y_train_labels = np.argmax(y_train, axis=1)
        acc = accuracy_score(y_train_labels, y_train_pred_labels)
        train_accuracies.append(acc)
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {acc*100:.2f}%")

    y_test_pred = mlp.forward(X_test)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    accuracy = accuracy_score(y_test_labels, y_test_pred_labels)
    print(f"\nAcurácia do Teste (Exercício 3): {accuracy * 100:.2f}%\n")

    # Plot training accuracy over epochs
    plt.figure(figsize=(8,4))
    plt.plot(train_accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'mlp','exercise3_accuracy.png'))
    plt.close()

    pca = PCA(n_components=2)
    X_test_2d = pca.fit_transform(X_test)

    # Plota a fronteira de decisão no espaço PCA
    plot_decision_boundary_pca(mlp, X_test, y_test, pca, 'exercise3_decision_boundary_pca.png')


def exercise_4():
    print("\n--- Executando Exercício 4: Classificação Multi-classe (2 camadas) ---")
    X_train, X_test, y_train, y_test, y_test_labels = generate_multiclass_data()

    input_size = X_train.shape[1]

    # hidden_sizes = [16, 8]
    # output_size = 3
    # learning_rate = 0.05
    # epochs = 500

    hidden_sizes = [64, 32] 
    output_size = 3
    learning_rate = 0.05
    epochs = 2000

    mlp = MLP(input_size, hidden_sizes, output_size, learning_rate, task='multiclass')
    train_accuracies = []
    for epoch in range(epochs):
        mlp.train(X_train, y_train, 1)
        y_train_pred = mlp.forward(X_train)
        y_train_pred_labels = np.argmax(y_train_pred, axis=1)
        y_train_labels = np.argmax(y_train, axis=1)
        acc = accuracy_score(y_train_labels, y_train_pred_labels)
        train_accuracies.append(acc)
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {acc*100:.2f}%")

    y_test_pred = mlp.forward(X_test)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    accuracy = accuracy_score(y_test_labels, y_test_pred_labels)
    print(f"\nAcurácia do Teste (Exercício 4): {accuracy * 100:.2f}%\n")

    # Plot training accuracy over epochs
    plt.figure(figsize=(8,4))
    plt.plot(train_accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'mlp','exercise4_accuracy.png'))
    plt.close()

    pca = PCA(n_components=2)
    X_test_2d = pca.fit_transform(X_test)

    # Plota a fronteira de decisão no espaço PCA
    plot_decision_boundary_pca(mlp, X_test, y_test, pca, 'exercise4_decision_boundary_pca.png')

def main():
    np.random.seed(42)
    
    exercise_2()
    exercise_3()
    exercise_4()

if __name__ == "__main__":
    main()