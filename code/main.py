import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from KNN_algorithm import KnnModel
from Visualization import dataVisualization


print('KNN algorithm from scratch...')
print('---------------------------------')

features, labels = load_digits(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state = 45, test_size = 0.2)

knn_model_scratch = KnnModel(3, x_train, y_train)
predicted_values = knn_model_scratch.predict(x_test) 

accuracy = accuracy_score(y_test, predicted_values)

print(f'Accuracy: {accuracy}')
print('confusion matrix')
print(confusion_matrix(y_true = y_test, y_pred = predicted_values))
dataVisualization(x_test, y_test, predicted_values)

print('------------------------------------')
print("Done")

print('KNN algorithm on sklearn package')
print('---------------------------------')

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(x_train, y_train)
predicted_values = knn_model.predict(x_test)

accuracy = accuracy_score(y_test, predicted_values)
print(f'Accuracy: {accuracy}')

plot_confusion_matrix(knn_model, x_test, y_test)
plt.show()

dataVisualization(x_test, y_test, predicted_values)
print('-----------------------------------------')
print('Done')