

```markdown
# Practical 2: Implement Stemming

**Aim:** To implement stemming algorithms for text pre-processing using the NLTK library.

## Part 1: Basic Stemming Implementation

This script demonstrates stemming on a simple list of sentences using the `PorterStemmer`.

```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# ==========================================
# Step 1: Setup
# ==========================================
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# ==========================================
# Step 2: Load Data
# ==========================================
data = {'messages': [
    "The cats are playing in the garden.",
    "He is running quickly to catch the bus.",
    "The boys are enjoying their game.",
    "She was reading a book.",
    "I love to eat apples and bananas."
]}
df = pd.DataFrame(data)

# ==========================================
# Step 3: Initialize Tools
# ==========================================
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ==========================================
# Step 4: Define Pre-processing Function
# ==========================================
def preprocess_text(text):
    # 1. Tokenization
    tokens = word_tokenize(text)
    
    # 2. Stop-words Removal
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # 3. Text Normalization (Lowercasing and Removing punctuation)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    
    # 4. Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# ==========================================
# Step 5: Execute and Display
# ==========================================
# Apply the pre-processing pipeline
df['processed_messages'] = df['messages'].apply(preprocess_text)

# Display the result
print(df[['messages', 'processed_messages']])
```

**Output:**
```text
                                  messages        processed_messages
0      The cats are playing in the garden.           cat play garden
1  He is running quickly to catch the bus.  run quickli catch bu
2        The boys are enjoying their game.            boy enjoy game
3                  She was reading a book.                 read book
4        I love to eat apples and bananas.  love eat appl banana
```
*,,,*

---

## Part 2: Stemming on SMS Dataset

This script applies the same stemming pipeline to the SMS dataset used in Practical 1.

```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# ==========================================
# Step 1: Setup and Load Data
# ==========================================
nltk.download('punkt')
nltk.download('stopwords')

# Sample data representing the SMS Corpus
data = {'messages': [
    """ham        Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...""",
    """ham        Ok lar... Joking wif u oni...""",
    """spam       Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's""",
    """ham        U dun say so early hor... U c already then say...""",
    """ham        Nah I don't think he goes to usf, he lives around here though"""
]}
df = pd.DataFrame(data)

# ==========================================
# Step 2: Define Pipeline
# ==========================================
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Stop-words Removal
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Normalization
    tokens = [word.lower() for word in tokens if word.isalnum()]
    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# ==========================================
# Step 3: Execute
# ==========================================
df['processed_messages'] = df['messages'].apply(preprocess_text)
print(df[['messages', 'processed_messages']])
```
*,,,*

---

## Report: Summary of Stemming Techniques

### 1. Available Stemmers
*   **Porter Stemmer:** A widely used, rule-based stemmer. It is efficient and simple, employing a set of rules to remove common English suffixes,.
*   **Snowball Stemmer (Porter2):** An improved version of the Porter Stemmer. It allows for more aggressive stemming and supports multiple languages.
*   **Lancaster Stemmer:** A highly aggressive algorithm that often reduces words to very short stems (sometimes not actual words) to drastically reduce word variations,.
*   **Regexp Stemmer:** Uses regular expressions to remove suffixes based on user-defined patterns, offering high flexibility.

### 2. Effectiveness and Comparison
*   **Porter Stemmer:** Moderate aggressiveness; implemented in NLTK as `PorterStemmer`,.
*   **Snowball Stemmer:** Refines Porter's rules and is generally more robust; implemented as `SnowballStemmer`.
*   **Lancaster Stemmer:** Significantly more aggressive than Porter or Snowball; useful for reducing word variations drastically but may result in stems that are difficult to interpret.
*   **Implementation:** All these stemmers are available via the `nltk.stem` module in Python. The choice of stemmer depends on the desired level of reduction and the specific requirements of the text analysis task.
```