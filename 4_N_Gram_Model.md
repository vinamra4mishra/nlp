Based on the **NLP_Zeeshan_Final.pdf** source, here is the Markdown file for **Practical 4**, which focuses on implementing N-gram models (Unigram and Bigram).

```markdown
# Practical 4: Implement N-gram Model

**Aim:** To implement Unigram and N-gram (specifically Bigram) models for text processing and prediction using Python and NLTK.

## Part 1: Unigram Model Implementation

The Unigram model calculates word frequencies and predicts the most common words in the corpus regardless of context.

```python
import nltk
from collections import defaultdict, Counter

# ==========================================
# Step 1: Setup
# ==========================================
# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('punkt_tab')

# ==========================================
# Step 2: Define Unigram Model Class
# ==========================================
import nltk
from collections import defaultdict, Counter

# ==========================================
# Step 1: Setup
# ==========================================
# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('punkt_tab')

# ==========================================
# Step 2: Define Unigram Model Class
# ==========================================
class UnigramModel:
    def __init__(self):
        self.word_freq = Counter()

    def preprocess(self, text):
        """Tokenizes text, converts to lowercase, and removes punctuation."""
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]
        return tokens

    def fit(self, text):
        """Builds the model by counting word frequencies."""
        tokens = self.preprocess(text)
        self.word_freq.update(tokens)

        # Debug: Print word frequencies
        print("Word frequencies:")
        for word, freq in self.word_freq.items():
            print(f"{word}: {freq}")

    def predict(self, word):
        """Returns the top 3 most frequent words in the corpus."""
        # Note: Unigram ignores the input 'word' context
        most_common = self.word_freq.most_common(3)
        return most_common

# ==========================================
# Step 3: Execution
# ==========================================
if __name__ == "__main__":
    # Sample text corpus
    corpus = """
    Natural language processing (NLP) is a subfield of artificial intelligence (AI).
    It enables computers to understand, interpret, and generate human language.
    With advances in machine learning and deep learning, NLP has made significant strides.
    Applications include sentiment analysis, machine translation, and chatbot development.
    """

    # Create and fit unigram model
    unigram_model = UnigramModel()
    unigram_model.fit(corpus)

    # Make predictions
    print("\nTop 3 predicted words based on frequency:")
    predictions = unigram_model.predict('any_word')
    print(predictions)
```

**Output:**
```text
Word frequencies:
natural: 1
language: 2
processing: 1
nlp: 2
...
and: 3
...
Top 3 predicted words based on frequency:
[('and', 3), ('language', 2), ('nlp', 2)]
```
*,,,,*

---

## Part 2: Bigram (N-Gram) Model Implementation

The Bigram model predicts the next word based on the preceding word (prefix) by analyzing pairs of consecutive words.

```python
import nltk
from collections import defaultdict, Counter

# Ensure NLTK resources are available
nltk.download('punkt')

# ==========================================
# Step 1: Define N-Gram Model Class
# ==========================================
class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams_freq = defaultdict(Counter)

    def preprocess(self, text):
        """Tokenizes text, converts to lowercase, and removes punctuation."""
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]
        return tokens

    def fit_bigram(self, text):
        """Generates n-grams and counts frequencies of the next word."""
        tokens = self.preprocess(text)
        
        # Generate n-grams
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            # Store frequency: prefix -> next_word
            self.ngrams_freq[ngram[:-1]][ngram[-1]] += 1
            
            # Debug: Print the generated ngram
            print(f"Generated bigram: {ngram[:-1]} -> {ngram[-1]}")

    def predict_bigram(self, prefix):
        """Predicts the next word based on the input prefix."""
        prefix = self.preprocess(prefix)
        
        # Check if we have enough words for the prefix
        if len(prefix) < self.n - 1:
            return []
            
        # Get the last (n-1) words to use as the lookup key
        prefix_tuple = tuple(prefix[-(self.n - 1):])
        print(f"Checking prefix: {prefix_tuple}")
        
        if prefix_tuple in self.ngrams_freq:
            next_words = self.ngrams_freq[prefix_tuple]
            return next_words.most_common(3)  # Return top 3 predictions
        else:
            return []

# ==========================================
# Step 2: Execution
# ==========================================
if __name__ == "__main__":
    # Sample text corpus
    corpus = """
    Natural language processing (NLP) is a subfield of artificial intelligence (AI).
    It enables computers to understand, interpret, and generate human language.
    With advances in machine learning and deep learning, NLP has made significant strides.
    Applications include sentiment analysis, machine translation, and chatbot development.
    """

    # Create and fit bigram model (n=2)
    bigram_model = NGramModel(n=2)
    bigram_model.fit_bigram(corpus)

    # Make predictions for the phrase 'natural language'
    # The model will use the last word 'language' as the prefix
    print("\nBigram Predictions for 'natural language':")
    predictions = bigram_model.predict_bigram('natural language')
    print(predictions)
```

**Output:**
```text
Generated bigram: ('natural',) -> language
Generated bigram: ('language',) -> processing
...
Generated bigram: ('language',) -> with
...
Bigram Predictions for 'natural language':
Checking prefix: ('language',)
[('processing', 1), ('with', 1)]
```
*,,,,,,,*
```