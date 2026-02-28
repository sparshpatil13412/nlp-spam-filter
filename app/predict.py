import joblib
import os

# Load the saved model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "spam_classifier_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_spam(input_texts):
    # Transform the input texts using the loaded vectorizer
    transformed_texts = vectorizer.transform([input_texts])
    
    # Predict probabilities using the loaded model
    probs = model.predict_proba(transformed_texts)[:, 1]
    
    # Classify as spam if probability > 0.12
    predictions = (probs > 0.12).astype(int)
    
    return predictions, probs

if __name__ == "__main__":
    email = input("Enter the email:")
    prediction, prob = predict_spam(email)
    if prediction == 1:
        print("Spam Email")
    else:
        print("Not Spam Email")

    if prob > 0.7:
        print("Confidence: Very High")
    elif prob > 0.4:
        print("Confidence: Medium")
    else:
        print("Confidence: Low")