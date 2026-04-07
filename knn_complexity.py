from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# Load dataset (Replace with your own X, y if available)
data = load_iris()
X = data.data
y = data.target

# 1. Splitting data to detect overfitting/underfitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Testing different K values
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Using cross-validation for stable accuracy estimate
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

# 3. Visualizing the 'Sweet Spot'
plt.figure(figsize=(8, 5))
plt.plot(k_range, cv_scores, marker='o',color = 'red')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding Optimal K (Bias–Variance Tradeoff)')
plt.grid(True)
plt.show()