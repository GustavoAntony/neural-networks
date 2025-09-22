## Manual Calculation of MLP Steps

This page details the step-by-step manual calculation of a single forward and backward pass for a simple Multi-Layer Perceptron (MLP). The network consists of two input features, one hidden layer with two neurons, and one output neuron.

## Network and Data Specifications

* **Input Vector ($x$)**: \([0.5, -0.2]\)
* **True Output ($y$)**: \(1.0\)
* **Hidden Layer Weights ($W^{(1)}$)**: \(\begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix}\)
* **Hidden Layer Biases ($b^{(1)}$)**: \([0.1, -0.2]\)
* **Output Layer Weights ($W^{(2)}$)**: \([0.5, -0.3]\)
* **Output Layer Bias ($b^{(2)}$)**: \(0.2\)
* **Activation Function**: Hyperbolic tangent (tanh) for both layers.
* **Loss Function**: Mean Squared Error (MSE), \(L = \frac{1}{N}(y - \hat{y})^2\).
* **Learning Rate ($\eta$)**: \(0.1\)

---

### 1. Forward Pass

The forward pass is the process of moving the input data through the network layers to produce an output.

#### Hidden Layer Pre-activation ($z^{(1)}$)

$$
z^{(1)} =
\begin{bmatrix}
(0.3)(0.5) + (-0.1)(-0.2) \\
(0.2)(0.5) + (0.4)(-0.2)
\end{bmatrix}
+
\begin{bmatrix}
0.1 \\
-0.2
\end{bmatrix}
=
\begin{bmatrix}
0.15 + 0.02 \\
0.10 - 0.08
\end{bmatrix}
+
\begin{bmatrix}
0.1 \\
-0.2
\end{bmatrix}
=
\begin{bmatrix}
0.17 \\
0.02
\end{bmatrix}
+
\begin{bmatrix}
0.1 \\
-0.2
\end{bmatrix}
=
\begin{bmatrix}
0.27 \\
-0.18
\end{bmatrix}
$$

#### Hidden Layer Activations ($a^{(1)}$)

$$
a^{(1)} =
\tanh\big(z^{(1)}\big) =
\begin{bmatrix}
\tanh(0.27) \\
\tanh(-0.18)
\end{bmatrix}
\approx
\begin{bmatrix}
0.2636 \\
-0.1781
\end{bmatrix}
$$

#### Output Layer Pre-activation ($z^{(2)}$)

$$
z^{(2)} =
\begin{bmatrix} 0.5 & -0.3 \end{bmatrix}
\begin{bmatrix} 0.2636 \\ -0.1781 \end{bmatrix}
+ 0.2
= (0.5)(0.2636) + (-0.3)(-0.1781) + 0.2
= 0.1318 + 0.0534 + 0.2
\approx 0.3852
$$

#### Final Output ($\hat{y}$)

$$
\hat{y} = \tanh(z^{(2)}) = \tanh(0.3852) \approx 0.3672
$$

---

### 2. Loss Calculation

$$
L = (y - \hat{y})^2 = (1.0 - 0.3672)^2 = (0.6328)^2 \approx 0.4004
$$

---

### 3. Backward Pass (Backpropagation)

#### Gradient of Loss w.r.t. Output Pre-activation

$$
\frac{\partial L}{\partial z^{(2)}} =
[-2(1 - \hat{y})]\cdot(1 - \hat{y}^2)
= [-2(0.6328)]\cdot(1 - 0.1348)
= -1.2656 \cdot 0.8652
\approx -1.0948
$$

---

#### Gradients for the Output Layer \((W^{(2)}, b^{(2)})\)

$$
\frac{\partial L}{\partial W^{(2)}} =
\frac{\partial L}{\partial z^{(2)}} \, a^{(1)T}
= (-1.0948)\cdot\begin{bmatrix}0.2636 & -0.1781\end{bmatrix}
\approx \begin{bmatrix}-0.2886 & 0.1950\end{bmatrix}
$$

$$
\frac{\partial L}{\partial b^{(2)}} = -1.0948
$$

---

#### Propagating to the Hidden Layer

$$
\frac{\partial L}{\partial a^{(1)}} =
W^{(2)T}\cdot\frac{\partial L}{\partial z^{(2)}}
= \begin{bmatrix}0.5\\ -0.3\end{bmatrix}\cdot(-1.0948)
\approx \begin{bmatrix}-0.5474 \\ 0.3284\end{bmatrix}
$$

$$
1 - \tanh^2(z^{(1)}) \approx
\begin{bmatrix}1 - (0.2636)^2 \\ 1 - (-0.1781)^2\end{bmatrix}
\approx
\begin{bmatrix}0.9306 \\ 0.9683\end{bmatrix}
$$

$$
\frac{\partial L}{\partial z^{(1)}} =
\begin{bmatrix}-0.5474 \\ 0.3284\end{bmatrix}
\odot
\begin{bmatrix}0.9306 \\ 0.9683\end{bmatrix}
\approx
\begin{bmatrix}-0.5094 \\ 0.3180\end{bmatrix}
$$

---

#### Gradients for the Hidden Layer \((W^{(1)}, b^{(1)})\)

$$
\frac{\partial L}{\partial W^{(1)}} =
\frac{\partial L}{\partial z^{(1)}} x^T =
\begin{bmatrix}-0.5094 \\ 0.3180\end{bmatrix}
\begin{bmatrix}0.5 & -0.2\end{bmatrix}
\approx
\begin{bmatrix}
-0.2547 & 0.1019 \\
0.1590 & -0.0636
\end{bmatrix}
$$

$$
\frac{\partial L}{\partial b^{(1)}} =
\begin{bmatrix}-0.5094 \\ 0.3180\end{bmatrix}
$$

---

### 4. Parameter Update

#### Output Layer

* **Weights**

$$
W^{(2)}_{new} =
\begin{bmatrix}0.5 & -0.3\end{bmatrix}
- 0.1\cdot\begin{bmatrix}-0.2886 & 0.1950\end{bmatrix}
\approx
\begin{bmatrix}0.5289 & -0.3195\end{bmatrix}
$$

* **Bias**

$$
b^{(2)}_{new} = 0.2 - 0.1\cdot(-1.0948) \approx 0.3095
$$

---

#### Hidden Layer

* **Weights**

$$
W^{(1)}_{new} =
\begin{bmatrix}
0.3 & -0.1 \\
0.2 & 0.4
\end{bmatrix}
+ (-0.1)\cdot
\begin{bmatrix}
-0.2547 & 0.1019 \\
0.1590 & -0.0636
\end{bmatrix}
\approx
\begin{bmatrix}
0.3255 & -0.1102 \\
0.1841 & 0.4064
\end{bmatrix}
$$

* **Biases**

$$
b^{(1)}_{new} =
\begin{bmatrix}0.1 \\ -0.2\end{bmatrix}
- 0.1\cdot
\begin{bmatrix}-0.5094 \\ 0.3180\end{bmatrix}
\approx
\begin{bmatrix}0.1509 \\ -0.2318\end{bmatrix}
$$

---
***The creation of this page was assisted by Google Gemini.**