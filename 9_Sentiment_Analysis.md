# Practical 9: Implement Sentiment Analysis

**Aim:** To implement a Sentiment Analysis model that classifies text reviews as positive or negative using the Naive Bayes algorithm.

## Python Implementation

This script uses `pandas` for data handling, `nltk` for text cleaning, and `scikit-learn` for vectorization and model training.

```python
# ==========================================
# Step 1: Setup and Installation
# ==========================================
# Install necessary libraries if not present
# !pip install pandas nltk scikit-learn

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# ==========================================
# Step 2: Load Data
# ==========================================
# Load the dataset (Ensure 'TestLarge.csv' is in the specified path)
try:
    df = pd.read_csv('/content/TestLarge.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the path.")
    # Creating a dummy dataset for demonstration if file is missing
    data = {
        'Review': [
            "This product is amazing!", "I hate this item.", "Best purchase ever.", 
            "Terrible quality.", "Highly recommended.", "Waste of money."
        ],
        'Sentiment': [
            "Positive", "Negative", "Positive", 
            "Negative", "Positive", "Negative"
        ]
    }
    df = pd.DataFrame(data)

# ==========================================
# Step 3: Data Pre-processing
# ==========================================
def preprocess_text(text):
    """
    Cleans text by:
    1. Converting to lowercase
    2. Tokenizing
    3. Removing non-alphabetic tokens and stop-words
    """
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing to the 'Review' column
df['Processed_Review'] = df['Review'].apply(preprocess_text)

# ==========================================
# Step 4: Model Training
# ==========================================
# Define Features (X) and Labels (y)
X = df['Processed_Review']
y = df['Sentiment']

# Split data into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization (Convert text to numerical counts)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# ==========================================
# Step 5: Evaluation
# ==========================================
# Predict on the test set
y_pred = model.predict(X_test_vectorized)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
# Note: pos_label='Positive' is specific to the target labels in the dataset
precision = precision_score(y_test, y_pred, pos_label='Positive', average='binary')
recall = recall_score(y_test, y_pred, pos_label='Positive', average='binary')

print(f'\nModel Evaluation:')
print(f'Accuracy:  {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall:    {recall:.2f}')

# ==========================================
# Step 6: Prediction on New Data
# ==========================================
# Define sample reviews for testing
sample_reviews = [
    "This product is amazing!",
    "I'm really disappointed with my purchase.",
    "It does exactly what I needed.",
    "Not worth the price.",
    "I would buy this again in a heartbeat!"
]

# Preprocess and vectorize samples
sample_reviews_processed = [preprocess_text(review) for review in sample_reviews]
sample_reviews_vectorized = vectorizer.transform(sample_reviews_processed)

# Predict sentiment
sample_predictions = model.predict(sample_reviews_vectorized)

# Display results
print("\nSample Predictions:")
for review, sentiment in zip(sample_reviews, sample_predictions):
    print(f'Review: "{review}" - Predicted Sentiment: {sentiment}')
```

**Output:**
```text
Model Evaluation:
Accuracy:  0.95
Precision: 0.90
Recall:    1.00

Sample Predictions:
Review: "This product is amazing!" - Predicted Sentiment: Positive
Review: "I'm really disappointed with my purchase." - Predicted Sentiment: Negative
Review: "It does exactly what I needed." - Predicted Sentiment: Positive
Review: "Not worth the price." - Predicted Sentiment: Positive
Review: "I would buy this again in a heartbeat!" - Predicted Sentiment: Negative
```
*,,,,,*
```