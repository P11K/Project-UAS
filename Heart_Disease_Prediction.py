import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

# Read and Analyse Data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("heart.csv")
df.head()
df.info()
df.columns
df.tail()
df.describe().T
df.isnull().sum()

# Missing Value Analysis
df.isnull().sum().sum()

# Data Visualization (Graphical components removed)

categorical_list = ["sex", "cp","fbs","restecg","exng","slp","caa","thall","output"]

df["age"].value_counts().plot.barh();
df["caa"].value_counts().plot.barh();
df["sex"].value_counts().plot.barh();
df["oldpeak"].value_counts().plot.barh();

df_num = df.select_dtypes(include=["float64", "int64"])
df_num.head()
df_num.describe().T
df_num["thall"].describe()

print("Mean: " + str(df_num["thall"].mean()))
print("Count: " + str(df_num["thall"].count())) 
print("Max: " + str(df_num["thall"].max()))
print("Min: " + str(df_num["thall"].min()))
print("Median: " + str(df_num["thall"].median()))
print("Standard Deviation: " + str(df_num["thall"].std()))

df["cp"].value_counts()
df["chol"].value_counts()
df["thall"].value_counts().plot.barh().set_title("thall class ");

categorical_list = ["sex", "cp","fbs","restecg","exng","slp","caa","thall","output"]

df_categoric = df.loc[:, categorical_list]
for i in categorical_list:
    print(i)
    print(df_categoric[i].value_counts())

numeric_list = ["age", "trtbps","chol","thalachh","oldpeak","output"]
df_numeric = df.loc[:, numeric_list]

# Standardization
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df[numeric_list[:-1]])
df_dummy = pd.DataFrame(scaled_array, columns=numeric_list[:-1])
df_dummy = pd.concat([df_dummy, df.loc[:, "output"]], axis=1)

# Box plot
data_melted = pd.melt(df_dummy, id_vars="output", var_name="features", value_name="value")

# Correlation Analysis
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), annot=True, fmt=".1f", linewidths=.7)
plt.show()

numeric_list = ["age", "trtbps","chol","thalachh","oldpeak"]
df_numeric = df.loc[:, numeric_list]
df_numeric.head()

# Outlier detection
for i in numeric_list:
    Q1 = np.percentile(df.loc[:, i],25)
    Q3 = np.percentile(df.loc[:, i],75)
    IQR = Q3 - Q1
    upper = np.where(df.loc[:, i] >= (Q3 +2.5*IQR))
    lower = np.where(df.loc[:, i] <= (Q1 - 2.5*IQR))
    try:
        df.drop(upper[0], inplace=True)
    except: 
        print("KeyError: {} not found in axis".format(upper[0]))
    try:
        df.drop(lower[0], inplace=True)
    except:  
        print("KeyError: {} not found in axis".format(lower[0]))

df1 = df.copy()

# Encoding Categorical Columns
df1 = pd.get_dummies(df1, columns=categorical_list[:-1], drop_first=True)
df1.head()

X = df1.drop(["output"], axis=1)
y = df1[["output"]]

scaler = StandardScaler()
X[numeric_list[:-1]] = scaler.fit_transform(X[numeric_list[:-1]])
X.head()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)
print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("y_train: {}".format(y_train.shape))
print("y_test: {}".format(y_test.shape))

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
print("Test accuracy: {}".format(accuracy_score(y_pred, y_test)))

# SVM
svm_model = SVC(kernel="linear").fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred)

# KNeighborsClassifier
df1 = df.copy()
y = df1["output"]
X = df1.drop(['output'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)

print("best score:" + str(knn_cv.best_score_))
print("best parameter: " + str(knn_cv.best_params_))

knn = KNeighborsClassifier(11)
knn_tuned = knn.fit(X_train, y_train)
print(knn_tuned.score(X_test, y_test))
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

# Close all figures
plt.close('all')
import pickle 
with open ('nyubo_pickle','wb') as r:
    pickle.dump(knn_model,r)