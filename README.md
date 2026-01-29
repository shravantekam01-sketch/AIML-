import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Dataset (Assuming a CSV with 'text' and 'label' columns)
# For this example, let's imagine a simple dictionary
data = {
    'text': [
        'Get a free gift card now!', 'Hey, are we meeting for lunch?',
        'WINNER! Claim your prize.', 'Please review the attached invoice.',
        'Urgent: Your account is locked.', 'Can you send me the notes from class?'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)

# 2. Preprocessing & Vectorization
# TF-IDF converts text to a matrix of importance scores
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['text'])
y = df['label']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model (Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Predict and Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
Step 10: Predict Price for New House
new_house = [[2000, 3, 2, 5]]
predicted_price = model.predict(new_house)
print("Predicted Price for New House:", predicted_price[0])
