import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class SpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.model = MultinomialNB()
        self.ps = PorterStemmer()
        self.is_trained = False
        
        # Download NLTK data if not present
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        words = text.split()
        # Remove stopwords and apply stemming
        stop_words = set(stopwords.words('english'))
        words = [self.ps.stem(word) for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    
    def train(self, texts, labels):
        """Train the model with given texts and labels"""
        print("Cleaning training texts...")
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        print("Vectorizing texts...")
        X = self.vectorizer.fit_transform(cleaned_texts)
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        return accuracy
    
    def predict(self, text):
        """Predict if text is spam or ham"""
        if not self.is_trained:
            raise Exception("Model not trained yet! Please train the model first.")
        
        cleaned_text = self.clean_text(text)
        text_vector = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0]
        
        return prediction, probability
    
    def save_model(self, filename):
        """Save the trained model to file"""
        if not self.is_trained:
            raise Exception("No trained model to save!")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a trained model from file"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")
            
        model_data = joblib.load(filename)
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filename}")
