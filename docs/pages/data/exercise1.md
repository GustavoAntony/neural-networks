## 1. Generate the Data

``` py
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
```

We're generating a synthetic dataset to simulate a classification problem. The data consists of four distinct classes, each following a bivariate normal distribution. This is a common practice for creating simple, interpretable datasets to test machine learning algorithms like K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Neural Networks.

We'll use a dictionary to define the parameters for each class: its mean (center) and standard deviation (spread). For each class, we generate 100 data points. The X array stores the 2D coordinates of these points, and the y array stores their corresponding class labels.

#### Key Concepts:

- **Bivariate Normal Distribution**: A probability distribution that describes two correlated variables. In our case, the two variables are the x and y coordinates of our data points. The mean parameter defines the center of the cluster, and the std parameter defines how spread out the data points are along each dimension.

- **np.random.normal()**: This NumPy function is used to generate random numbers from a normal (Gaussian) distribution. We use it to create our data points.

- **np.vstack() and np.hstack()**: These functions combine the generated points and labels from each class into single, continuous NumPy arrays.

## 2. Plot the data

``` py
colors = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(8, 6))
for cls in params.keys():
    plt.scatter(X[y == cls, 0], X[y == cls, 1], c=colors[cls], label=f"Class {cls}", alpha=0.6)

plt.xlabel("Label 1")
plt.ylabel("Label 2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'data','exercise1_2.png'))
```

![Scatter Plot](../../assets/images/data/exercise1_2.png)

After generating the data, a scatter plot is the ideal way to visualize the distribution of each class. This visual representation helps us understand the data's characteristics, identify patterns, and gauge how separable the classes are.

## 3. Analyze and Draw Boundaries:

``` py
colors = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(8, 6))
for cls in params.keys():
    plt.scatter(X[y == cls, 0], X[y == cls, 1], c=colors[cls], label=f"Class {cls}", alpha=0.6)

plt.xlabel("Label 1")
plt.ylabel("Label 2")
plt.legend()
plt.grid(True, alpha=0.3)

plt.plot([2.5, 11.5], [-3, 9], color='black', linestyle='--', linewidth=2, label="Boundary 0-1")
plt.plot([2.0, 4.5], [12.5, -0.2], color='black', linestyle='--', linewidth=2, label="Boundary 1-2")
plt.plot([11.5, 11.5], [-2, 14], color='black', linestyle='--', linewidth=2, label="Boundary 2-3")

plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'data','exercise1_3.png'))
```

![Scatter Plot](../../assets/images/data/exercise1_3.png)

With the data visualized, we can analyze the class separation. Our goal is to determine if a simple linear model could effectively classify the data or if a more complex model is required. We can visually represent potential decision boundaries to test this hypothesis.

The scatter plot shows that the clusters have some overlap, especially between classes 0, 1, and 2. Because of this overlap, a single straight line cannot perfectly separate all four classes. This is a classic example of a non-linearly separable dataset.

However, a combination of multiple linear boundaries can effectively separate the regions. This is what multi-layer neural networks do, they use a series of linear transformations to create complex, non-linear decision boundaries. The added lines in the plot below demonstrate that a series of linear splits can achieve near-perfect separation.

## Final Code: Integrated Solution

Here is the complete code for generating the data, visualizing it, and analyzing the boundaries:

``` py
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
```