from flask import Flask, render_template, request, jsonify
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import SpamDetector
    print("✓ Successfully imported SpamDetector")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure model.py is in the same directory")
    exit(1)

app = Flask(__name__)

# Initialize detector
detector = SpamDetector()
model_loaded = False

# Try to load the trained model
MODEL_FILE = 'spam_model.pkl'

if os.path.exists(MODEL_FILE):
    try:
        detector.load_model(MODEL_FILE)
        model_loaded = True
        print("✓ Model loaded successfully from spam_model.pkl")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run train_model.py first to create the model")
else:
    print(f"❌ Model file '{MODEL_FILE}' not found")
    print("Please run train_model.py first to train and save the model")

@app.route('/')
def home():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return render_template('result.html', 
                             prediction="Error", 
                             confidence=0,
                             message="Model not loaded. Please run train_model.py first.")
    
    try:
        message = request.form['message']
        
        if not message.strip():
            return render_template('result.html',
                                 prediction="Error",
                                 confidence=0,
                                 message="Please enter a message to analyze.")
        
        # Make prediction
        prediction, probabilities = detector.predict(message)
        
        # Convert prediction to human-readable format
        result = "SPAM" if prediction == 1 else "HAM"
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        confidence_percent = round(confidence * 100, 2)
        
        return render_template('result.html',
                             prediction=result,
                             confidence=confidence_percent,
                             message=message)
    
    except Exception as e:
        return render_template('result.html',
                             prediction="Error",
                             confidence=0,
                             message=f"An error occurred: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message']
        
        if not message.strip():
            return jsonify({'error': 'Empty message'}), 400
        
        prediction, probabilities = detector.predict(message)
        
        return jsonify({
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': float(probabilities[1] if prediction == 1 else probabilities[0]),
            'message': message
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    print(f"\n🚀 Starting Flask app...")
    print(f"📊 Model loaded: {model_loaded}")
    print(f"🌐 Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)