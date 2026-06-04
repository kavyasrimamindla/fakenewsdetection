import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt
import io, base64

# ---------------- LOAD DATA ----------------
fake = pd.read_csv("Fake.csv", encoding="latin1")
true = pd.read_csv("True.csv", encoding="latin1")

# Label them
fake['label'] = 0
true['label'] = 1

# Combine original datasets
data = pd.concat([fake[['text', 'label']], true[['text', 'label']]], axis=0)

# Load the new improved dataset
extra = pd.read_csv("extra_data.csv")
data = pd.concat([data, extra], axis=0)

# Drop empty or NaN rows
data.dropna(subset=['text'], inplace=True)

# ---------------- TRAIN MODEL ----------------
x = data['text']
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(xv_train, y_train)

y_pred = model.predict(xv_test)

# ---------------- METRICS ----------------
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# ---------------- PIE CHART ----------------
labels = ['Fake', 'Real']
sizes = [sum(y_pred == 0), sum(y_pred == 1)]
colors = ['#e74c3c', '#2ecc71']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Fake vs Real News Distribution (Predicted)')
plt.axis('equal')
plt.show()
