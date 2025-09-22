## MLP Implementation

Here is a simple implementation of a Multi-Layer Perceptron (MLP) in Python using NumPy. This implementation includes forward and backward propagation, as well as parameter updates using gradient descent. The MLP can be configured for binary or multi-class classification tasks.

```py
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, task='binary'):
        self.lr = learning_rate
        self.task = task
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layers = []  # Lista para armazenar pesos e vieses de cada camada
        
        # Criação das camadas da rede
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'W': W, 'b': b, 'a': None, 'z': None}) # 'a' e 'z' armazenam ativações para o backprop

    # Funções de ativação
    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_derivative(self, a):
        return 1 - np.power(a, 2)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Funções de perda
    def _binary_cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -(1/m) * np.sum(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))
        return loss

    def _categorical_cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        return loss

    def forward(self, X):
        activations = [X]
        # Forward pass para todas as camadas ocultas
        for i in range(len(self.hidden_sizes)):
            z = np.dot(activations[i], self.layers[i]['W']) + self.layers[i]['b']
            a = self._tanh(z)
            self.layers[i]['z'] = z
            self.layers[i]['a'] = a
            activations.append(a)
        
        # Forward pass para a camada de saída
        z_out = np.dot(activations[-1], self.layers[-1]['W']) + self.layers[-1]['b']
        self.layers[-1]['z'] = z_out
        
        if self.task == 'binary':
            a_out = self._sigmoid(z_out)
        elif self.task == 'multiclass':
            a_out = self._softmax(z_out)
        self.layers[-1]['a'] = a_out
        
        return a_out

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        grads = {}
        
        # Gradiente para a camada de saída (a última camada na lista)
        dz = (y_pred - y_true) / m
        grads[len(self.layers)-1] = {
            'dW': np.dot(self.layers[-2]['a'].T if len(self.layers) > 1 else X.T, dz),
            'db': np.sum(dz, axis=0, keepdims=True)
        }
        
        # Backward pass para as camadas ocultas (loop de trás para frente)
        for i in range(len(self.layers) - 2, -1, -1):
            da = np.dot(dz, self.layers[i+1]['W'].T)
            dz = da * self._tanh_derivative(self.layers[i]['a'])
            
            # O input para a primeira camada oculta é X
            input_a = self.layers[i-1]['a'] if i > 0 else X
            
            grads[i] = {
                'dW': np.dot(input_a.T, dz),
                'db': np.sum(dz, axis=0, keepdims=True)
            }
        
        return grads

    def update_parameters(self, grads):
        for i in range(len(self.layers)):
            self.layers[i]['W'] -= self.lr * grads[i]['dW']
            self.layers[i]['b'] -= self.lr * grads[i]['db']
            
    def train(self, X_train, y_train, epochs):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            
            if self.task == 'binary':
                loss = self._binary_cross_entropy(y_train, y_pred)
            elif self.task == 'multiclass':
                loss = self._categorical_cross_entropy(y_train, y_pred)
                
            grads = self.backward(X_train, y_train, y_pred)
            self.update_parameters(grads)
            losses.append(loss)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        return losses
```