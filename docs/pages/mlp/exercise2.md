This exercise demonstrates how to build and train a simple Multi-Layer Perceptron (MLP) for a binary classification problem using synthetic data. The process includes data generation, visualization, model setup, training, evaluation, and visualization of results.

## Data Generation

We generate two classes of synthetic data using `make_classification`, with some noise and class separation. The data is split into training and test sets.

```python
X_train, X_test, y_train, y_test = generate_binary_data()
```

## Data Visualization

Before training, it's important to visualize the data distribution. Here, we plot the training data to see the class separation in 2D.

```python
plt.scatter(X_train[:,0], X_train[:,1], c=y_train.flatten(), cmap='bwr', alpha=0.6)
plt.title('Training Data - Exercise 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'mlp','exercise2_data.png'))
plt.close()
```

![Training Data](../../assets/images/mlp/exercise2_data.png)

## Model Initialization

We define the MLP architecture: input size, one hidden layer, output size, and learning rate. The model is created for binary classification.

```python
input_size = X_train.shape[1]
hidden_sizes = [16]
output_size = 1
learning_rate = 0.05
mlp = MLP(input_size, hidden_sizes, output_size, learning_rate, task='binary')
```

## Training and Accuracy per Epoch

The model is trained for 2000 epochs. After each epoch, we compute and store the training accuracy to monitor learning progress.

```python
train_accuracies = []
epochs = 2000
for epoch in range(epochs):
    mlp.train(X_train, y_train, 1)
    y_train_pred = mlp.forward(X_train)
    y_train_pred_classes = (y_train_pred > 0.5).astype(int)
    acc = accuracy_score(y_train, y_train_pred_classes)
    train_accuracies.append(acc)
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {acc*100:.2f}%")
```

## Test Evaluation

After training, we evaluate the model on the test set to measure its generalization performance.

```python
y_test_pred = mlp.forward(X_test)
y_test_pred_classes = (y_test_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_test_pred_classes)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

**Test Accuracy: 82.50%**

## Training Accuracy Plot

The following plot shows how the training accuracy evolves over the epochs, helping to visualize convergence and possible overfitting.

```python
plt.figure(figsize=(8,4))
plt.plot(train_accuracies, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'mlp','exercise2_accuracy.png'))
plt.close()
```

![Training Accuracy](../../assets/images/mlp/exercise2_accuracy.png)

## Decision Boundary

Finally, we visualize the decision boundary learned by the MLP. This helps to understand how well the model separates the two classes in the feature space.

```python
if input_size == 2:
    plot_decision_boundary(mlp, X_test, y_test, 'exercise2_decision_boundary.png')
```

![Decision Boundary](../../assets/images/mlp/exercise2_decision_boundary.png)
