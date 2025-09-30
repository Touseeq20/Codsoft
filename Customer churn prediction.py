import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score, precision_score, recall_score, f1_score
df = pd.read_csv('Churn_Modelling.csv', header = 0)
df.head()
df.tail()
df.shape
df.size
df.describe()
df.info()
df.isnull().values.any()
df.columns
df.index
print(df["Geography"].unique())
print(df["Gender"].unique())
print(df["NumOfProducts"].unique())
print(df["HasCrCard"].unique())
print(df["IsActiveMember"].unique())
print(df["Exited"].unique())
df.isnull().sum()
df.axes
df.iloc[0:4]
df.loc[:, "CustomerId"]
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()

plt.figure(figsize=(14,10))
sns.countplot(x='Geography', data=df,color="orange")
plt.xlabel('Geography Distribution')
plt.ylabel('Count')
plt.title('Geography Distribution Plot',fontsize=14, fontweight="bold", color = "green")
plt.show()

df.head()
df.drop(labels = ["RowNumber","CustomerId", "Surname"], axis = 1, inplace=True)
df = pd.get_dummies(df, drop_first = True)
df.head()
X = df.drop("Exited", axis = 1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
lr_y_pred = lr_classifier.predict(X_test)
print("Logistic Regression Model:")
print(confusion_matrix(y_test, lr_y_pred))
print(classification_report(y_test, lr_y_pred))
print("Accuracy: ", accuracy_score(y_test, lr_y_pred))
print("r2_Score: ", r2_score(y_test, lr_y_pred))
print("Precision_score: ", precision_score(y_test, lr_y_pred))
print("Recall_score: ", recall_score(y_test, lr_y_pred))
print("f1_score: ", f1_score(y_test, lr_y_pred))

rf_classifier = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)
print("Random Forest Model:")
print(confusion_matrix(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))
print("Accuracy: ", accuracy_score(y_test, rf_y_pred))
print("r2_Score: ", r2_score(y_test, rf_y_pred))
print("Precision_score: ", precision_score(y_test, rf_y_pred))
print("Recall_score: ", recall_score(y_test, rf_y_pred))
print("f1_score: ", f1_score(y_test, rf_y_pred))

gb_classifier = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.02, max_depth=1, random_state=42)
gb_classifier.fit(X_train, y_train)
gb_y_pred = gb_classifier.predict(X_test)
print("Gradient Boosting Model:")
print(confusion_matrix(y_test, gb_y_pred))
print(classification_report(y_test, gb_y_pred))
print("Accuracy: ", accuracy_score(y_test, gb_y_pred))
print("r2_Score: ", r2_score(y_test, gb_y_pred))
print("Precision_score: ", precision_score(y_test, gb_y_pred))
print("Recall_score: ", recall_score(y_test, gb_y_pred))
print("f1_score: ", f1_score(y_test, gb_y_pred))
