## Generate the Data

``` py
params = {
    'A' : {MEAN : [0,0,0,0,0], COV : [[1,0.8,0.1,0,0],[0.8,1,0.3,0,0],[0.1,0.3,1,0.5,0],[0,0,0.5,1,0.2],[0,0,0,0.2,1.0]]},
    'B' : {MEAN : [1.5,1.5,1.5,1.5,1.5], COV : [[1.5,-0.7,0.2,0,0],[-0.7,1.5,0.4,0,0],[0.2,0.4,1.5,0.6,0],[0,0,0.6,1.5,0.3],[0,0,0,0.3,1.5]]}
}

cls_A = np.random.multivariate_normal(params['A'][MEAN], params['A'][COV], 500)
cls_B = np.random.multivariate_normal(params['B'][MEAN], params['B'][COV], 500)
```

We're creating a synthetic dataset with two distinct classes, A and B, in a 5-dimensional space. This is a common practice for creating complex, non-linear datasets that can't be easily visualized. Each class is defined by a multivariate normal distribution, a powerful way to model data clusters by specifying a central point (the mean vector) and the spread and relationship between its dimensions (the covariance matrix).

The covariance matrix is particularly important here. It's a 5x5 matrix that describes how each feature in the 5D space relates to the others. The diagonal elements represent the variance of each feature, while the off-diagonal elements show the covariance between them. A high positive or negative covariance indicates that two features are strongly related. For example, in class B, the negative covariance of -0.7 between the first two features indicates that as one increases, the other tends to decrease.

## Plot the Data

``` py
labels_A = np.zeros(500)
labels_B = np.ones(500)

X = np.vstack((cls_A, cls_B))
y = np.hstack((labels_A, labels_B))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```

Since our dataset exists in a 5D space, direct visualization is impossible. To analyze the data's structure, we use Principal Component Analysis (PCA) to reduce its dimensionality to 2D.

PCA works by identifying the directions (called principal components) that capture the most variance in the data. By projecting the high-dimensional data onto the first two principal components, we create a 2D representation that preserves as much of the original information as possible. This allows us to visualize the clusters and their relationships.

``` py
plt.figure(figsize=(8,6))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Classe A', alpha=0.7)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Classe B', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.savefig(os.path.join(IMAGES_OUTPUTS_FILE_PATH,'data','exercise2.png'))
plt.close()
```

![Scatter Plot](../../assets/images/data/exercise2.png)

The scatter plot of the PCA-reduced data reveals how the two classes are distributed in the lower-dimensional space. The plot shows that classes A and B have significant overlap. This is a key insight.

The overlap indicates that the data is not linearly separable. A simple linear classifier, like Logistic Regression or a single-layer perceptron, would struggle to find a single straight line to accurately separate the two classes. This type of non-linear problem is a challenge for simple models but is a perfect use case for more complex, non-linear models like Multi-Layer Perceptrons (Neural Networks), which can learn intricate decision boundaries.

## Final Code: Integrated Solution

Here is the complete code for generating the data, applying PCA, and visualizing the results:

``` py
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
```
