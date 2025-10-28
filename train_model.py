from model import SpamDetector

def create_training_data():
    """Create sample spam/ham training data"""
    spam_messages = [
        "WINNER!! You have won a $1000 gift card! Call now to claim.",
        "URGENT: Your account will be suspended. Click here to verify.",
        "Congratulations! You won a free iPhone. Reply to claim.",
        "Investment opportunity! Double your money in 24 hours.",
        "Free trial! Limited time offer. Click now!",
        "You have been selected for our exclusive membership.",
        "Your computer may be infected. Download protection now.",
        "Get rich quick with this amazing opportunity!",
        "Claim your prize! You are our lucky winner!",
        "Limited time discount! 90% off all products!",
        "You've been approved for a special loan offer!",
        "Your package delivery failed. Update your information now.",
        "Exclusive deal just for you! Don't miss out!",
        "Your account has suspicious activity. Verify immediately.",
        "Free gift card waiting for you! Claim now!",
        "You have unclaimed money waiting! Click to receive.",
        "Important security alert for your account!",
        "Special promotion: Buy one get one free!",
        "Your warranty is about to expire. Renew now!",
        "Earn money from home! No experience required!"
    ]
    
    ham_messages = [
        "Hey, are we meeting for lunch tomorrow?",
        "Your package will be delivered today between 2-4 PM.",
        "Don't forget the meeting at 3 PM today.",
        "Can you send me the report when you get a chance?",
        "Thanks for your help with the project.",
        "What time should I pick you up?",
        "The weather looks great for our trip this weekend.",
        "I'll be running about 10 minutes late.",
        "Do you want to grab coffee sometime this week?",
        "Don't forget to bring the documents tomorrow.",
        "Mom called and said she's coming over for dinner.",
        "Your doctor's appointment is confirmed for next Monday.",
        "The kids have soccer practice at 5 PM today.",
        "Can you pick up some milk on your way home?",
        "Looking forward to seeing you at the party!",
        "Happy birthday! Hope you have a great day!",
        "The project deadline has been extended to Friday.",
        "Could you please review this document?",
        "Let me know if you need any help with the presentation.",
        "What's your opinion on the new proposal?"
    ]
    
    # Combine messages and create labels
    messages = spam_messages + ham_messages
    labels = [1] * len(spam_messages) + [0] * len(ham_messages)
    
    return messages, labels

def main():
    print("Creating training data...")
    X, y = create_training_data()
    
    print(f"Training with {len(X)} messages ({sum(y)} spam, {len(y)-sum(y)} ham)")
    
    print("Initializing spam detector...")
    detector = SpamDetector()
    
    print("Training model...")
    accuracy = detector.train(X, y)
    
    print("Saving model...")
    detector.save_model('spam_model.pkl')
    
    print(f"\n🎯 Training completed! Model accuracy: {accuracy:.4f}")
    
    # Test the model with some examples
    print("\n🧪 Testing the model:")
    test_messages = [
        "Congratulations! You won a free iPhone!",
        "Hey, let's meet for coffee tomorrow",
        "URGENT: Your account will be suspended",
        "Can you send me the meeting notes?",
        "FREE MONEY!!! Click here now!!!"
    ]
    
    print("\n" + "="*50)
    for msg in test_messages:
        try:
            prediction, prob = detector.predict(msg)
            result = "SPAM" if prediction == 1 else "HAM"
            confidence = prob[1] if prediction == 1 else prob[0]
            print(f"📝 '{msg}'")
            print(f"   → {result} (confidence: {confidence:.2%})")
            print()
        except Exception as e:
            print(f"❌ Error predicting '{msg}': {e}")

if __name__ == "__main__":
    main()
