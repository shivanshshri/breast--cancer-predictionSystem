import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

print("--------------------------------------------------------------------------------------------\n")
print(data.head())
print("--------------------------------------------------------------------------------------------\n")
print(data.describe())
print("--------------------------------------------------------------------------------------------\n")
for col in data.columns:
    print(col)
print("--------------------------------------------------------------------------------------------\n")
print(data["diagnosis"].value_counts())
print("--------------------------------------------------------------------------------------------\n")
data = data.drop(["id"], axis = 1)
print(data.head())
print("--------------------------------------------------------------------------------------------\n")
print(data.corr)
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean, M.texture_mean, alpha = 0.3, color = "red", label = "M")
plt.scatter(B.radius_mean, B.texture_mean, color = "blue", alpha = 0.3, label = "B")
plt.legend()
plt.show()

data.diagnosis = [1 if i == "M" else 0 for i in data.diagnosis]

x = data.drop(["diagnosis"], axis=1)
y = data.diagnosis.values

# x = (x - np.min(x))/(np.max(x) - np.min(x))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X shape : ", x.shape)
print("X_test.shape : ", X_test.shape)
print("X_train shape : ", X_train.shape)


from sklearn.linear_model import LogisticRegression, LinearRegression

lg = LogisticRegression()

lg = lg.fit(X_train, y_train);

y_pred = lg.predict(X_test)

print(y_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1Score = f1_score(y_test, y_pred)
print("Accuracy : ", accuracy)
print("Precision : ", precision)
print("Recall : ", recall)
print("F1- Score : ", f1Score)


import pickle 
pickle.dump(lg, open('model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))