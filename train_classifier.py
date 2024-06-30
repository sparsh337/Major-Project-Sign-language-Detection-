import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()


#svm

# import pickle

# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# # Replace RandomForestClassifier with Support Vector Machine (SVM) classifier
# model = SVC()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly using SVM!'.format(score * 100))

# f = open('svm_model.p', 'wb')  # Change the filename to 'svm_model.p'
# pickle.dump({'model': model}, f)
# f.close()


#decision tree
# import pickle

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# # Replace RandomForestClassifier with Decision Tree classifier
# model = DecisionTreeClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly using Decision Tree!'.format(score * 100))

# f = open('decision_tree_model.p', 'wb')  # Change the filename to 'decision_tree_model.p'
# pickle.dump({'model': model}, f)
# f.close()

#knn

# import pickle
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# # Replace RandomForestClassifier with k-Nearest Neighbors classifier
# model = KNeighborsClassifier()
# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly using k-Nearest Neighbors!'.format(score * 100))

# # Using 'with' statement for file operations
# with open('knn_model.p', 'wb') as f:
#     pickle.dump({'model': model}, f)


#gradient boosting classifier

# import pickle
# from sklearn.ensemble import GradientBoostingClassifier  # Import Gradient Boosting Classifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# # Replace Support Vector Machine (SVM) with Gradient Boosting Classifier
# model = GradientBoostingClassifier()  # Change the model to GradientBoostingClassifier

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly using Gradient Boosting Classifier!'.format(score * 100))

# f = open('gradient_boosting_model.p', 'wb')  # Change the filename to 'gradient_boosting_model.p'
# pickle.dump({'model': model}, f)
# f.close()


#logistic regression

# import pickle
# from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# # Replace Support Vector Machine (SVM) with Logistic Regression
# model = LogisticRegression()  # Change the model to LogisticRegression

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly using Logistic Regression!'.format(score * 100))

# f = open('logistic_regression_model.p', 'wb')  # Change the filename to 'logistic_regression_model.p'
# pickle.dump({'model': model}, f)
# f.close()


#navy bassian

# import pickle
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import numpy as np

# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# # Replace Support Vector Machine (SVM) with Gaussian Naive Bayes
# model = GaussianNB()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# # Accuracy
# accuracy = accuracy_score(y_test, y_predict)
# print('Accuracy: {:.2f}%'.format(accuracy * 100))

# # Classification Report
# classification_rep = classification_report(y_test, y_predict)
# print('Classification Report:\n', classification_rep)

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_predict)
# print('Confusion Matrix:\n', conf_matrix)

# # You can extract precision, recall, and F1-score from the classification report if needed.