import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv(r'e:\Sparsh Python Programming folder\Spam AI\data\spam.csv', encoding='latin-1')

#Data Claning
mapping = {'ham': 0, 'spam': 1}
df['v1'] = df['v1'].map(mapping)

#Data Distribution
x = df["v1"]
y = df["v2"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Data Transformation
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)
y_train = vectorizer.fit_transform(y_train)
y_test = vectorizer.transform(y_test)

#Train Model
model = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1, 1: 2}
    )
model.fit(y_train, x_train)

#Predict Model
probs = model.predict_proba(y_test)[:, 1]
y_pred = (probs > 0.2).astype(int)

#Accuracy Check
accuracy = accuracy_score(x_test, y_pred)
cm = confusion_matrix(x_test, y_pred)
report = classification_report(x_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:{cm}")
print(f"Report:{report}")

#Visualization
precision, recall, thresholds = precision_recall_curve(x_test, probs)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# Save Model
joblib.dump(model, r'E:\Sparsh Python Programming folder\Spam AI\models\spam_classifier_model.pkl')
joblib.dump(vectorizer, r'E:\Sparsh Python Programming folder\Spam AI\models\tfidf_vectorizer.pkl')