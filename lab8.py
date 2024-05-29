from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
for i in range(len(iris_dataset.target_names)):
    print("Label ",i,"-", str(iris_dataset.target_names[i]))

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Result of classification using k-nn with k=1 ")
for i in range(len(X_test)):
    print("Sample:", X_test[i], "Actual label:", y_test[i], "Predicted label:", y_pred[i])

print("\nClassifier accuracy: ",classifier.score(X_test, y_test))
