from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

print("\nIRIS FEATURES / TARGET NAMES:\n", iris_dataset.target_names)
for i in range(len(iris_dataset.target_names)):
    print("[{0}]:[{1}]".format(i, iris_dataset.target_names[i]))

print("\nIRIS DATA:\n", iris_dataset["data"])

X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

print("\nTarget:\n", iris_dataset["target"])
print("\nX TRAIN\n", X_train)
print("\nX TEST\n", X_test)
print("\nY TRAIN\n", y_train)
print("\nY TEST\n", y_test)

kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(X_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
print("\nXNEW\n", x_new)
prediction = kn.predict(x_new)
print("\nPredicted target value: {}\n".format(prediction))
print("\nPredicted feature name: {}\n".format(iris_dataset["target_names"][prediction]))

for i in range(len(X_test)):
    x = X_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    print("\nActual: {0} {1}, Predicted: {2} {3}".format(
        y_test[i], iris_dataset["target_names"][y_test[i]], 
        prediction, iris_dataset["target_names"][prediction]))

print("\nTEST SCORE [ACCURACY]: {:.2f}\n".format(kn.score(X_test, y_test)))
