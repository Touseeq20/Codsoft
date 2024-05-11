import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder

# Load data
test_df = pd.read_csv('test_data.txt', sep=":::", header=0, engine='python')
train_df = pd.read_csv('train_data.txt', sep=":::", header=0, engine='python')

# Rename columns for clarity
train_df.columns = ['SN', 'movie_name', 'category', 'confession']
test_df.columns = ['SN', 'movie_name', 'confession']

# Data exploration
print(train_df.head())
print(test_df.head())
print(train_df.info())
print(test_df.info())
print(train_df.describe())
print(test_df.describe())

# Data visualization
plt.figure(figsize=(14, 10))
sns.countplot(x='category', data=train_df)
plt.xlabel('Movie Category')
plt.ylabel('Count')
plt.title('Movie Genre Plot')
plt.xticks(rotation=90)
plt.show()

# Combine train and test data for preprocessing
combined_df = pd.concat([train_df, test_df], axis=0)

# Label encode categorical variables
encoder = LabelEncoder()
combined_df['category'] = encoder.fit_transform(combined_df['category'])
combined_df['movie_name'] = encoder.fit_transform(combined_df['movie_name'])

# Fill missing values
combined_df['category'] = combined_df['category'].fillna(combined_df['category'].mean())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(combined_df['confession'])
y = combined_df['category']

# Split data back into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nAccuracy:", accuracy_score(y_test, predictions))

print("Naive Bayes Model:")
evaluate_model(nb_model, X_test, y_test)

print("\nLogistic Regression Model:")
evaluate_model(lr_model, X_test, y_test)
