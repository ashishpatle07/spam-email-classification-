# Import necessary libraries
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Using Logistic Regression instead of Naive Bayes
from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load Dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.rename(columns={"v1": "label", "v2": "message"})
data = data[['label', 'message']]
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Preprocess the text column in the data
data['cleaned_message'] = data['message'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['cleaned_message']).toarray()
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print("Accuracy on Test Set:", (y_pred == y_test).mean())  # Print accuracy

# Save the model and vectorizer using pickle
with open('spam_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Create a Flask app for the web application
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Load the model and vectorizer
    with open('spam_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Get the user input from the form
    user_input = request.form['email_text']
    
    # Preprocess the input text
    processed_input = preprocess_text(user_input)
    print("Processed Input:", processed_input)  # Debugging step
    
    # Vectorize the input text
    vectorized_input = vectorizer.transform([processed_input]).toarray()
    print("Vectorized Input Shape:", vectorized_input.shape)  # Debugging step
    
    # Make a prediction
    prediction = model.predict(vectorized_input)[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    return render_template('index.html', prediction=result)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
