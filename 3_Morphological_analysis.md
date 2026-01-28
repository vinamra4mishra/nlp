# Practical 3: Implement Morphological Analysis

**Aim:** To implement morphological analysis (lemmatization and morpheme extraction) using NLTK, SpaCy, and custom parsing logic.

## Part 1: Morphological Analysis using NLTK

This implementation uses the `WordNetLemmatizer` to reduce words to their base forms (lemmas).

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# ==========================================
# Step 1: Setup and Downloads
# ==========================================
# Download necessary NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ==========================================
# Step 2: Define Analysis Function
# ==========================================
def morphological_analysis(text):
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stop-words removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Text normalization (remove punctuation)
    normalized_tokens = [word.lower().strip(string.punctuation) for word in filtered_tokens]
    
    # Morphological analysis using lemmatization
    # 'pos=n' specifies the part of speech as noun
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos='n') for word in normalized_tokens]
    
    return lemmatized_tokens

# ==========================================
# Step 3: Execution
# ==========================================
text_sample = "The cats are playing with the ball and they are enjoying happiness"
processed_text = morphological_analysis(text_sample)
print(processed_text)
Output:
['cat', 'playing', 'ball', 'enjoying', 'happiness']
,,

--------------------------------------------------------------------------------
Part 2: Morphological Analysis using SpaCy
This implementation utilizes the spaCy library, which automatically handles part-of-speech tagging and lemmatization.
import spacy

# ==========================================
# Step 1: Load Model
# ==========================================
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# ==========================================
# Step 2: Define Analysis Function
# ==========================================
def spacy_morphological_analysis(word):
    # Process the word
    doc = nlp(word)
    # Extract text and lemma for each token
    analyzed_forms = [(token.text, token.lemma_) for token in doc]
    return analyzed_forms

# ==========================================
# Step 3: Execution
# ==========================================
word_sample = "playing"
analyzed_word = spacy_morphological_analysis(word_sample)
print(analyzed_word)
Output:
[('playing', 'play')]
,,

--------------------------------------------------------------------------------
Part 3: Custom Morphological Parser
This script implements a rule-based parser to break words down into prefixes, roots, and suffixes, alongside a basic Finite State Transducer (FST).
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# ==========================================
# Step 1: Define Morphological Rules
# ==========================================
def morphological_parsing(word):
    """
    Breaks a word into morphemes (prefix, root, suffix) based on rules.
    """
    morphemes = []
    
    # Check for 'un-' prefix and '-ness' suffix
    if word.startswith('un') and len(word) > 8:
        morphemes.append('un-')      # Prefix
        base_word = word[2:-4]       # Root (remove un- and -ness)
        morphemes.append(base_word)
        morphemes.append('-ness')    # Suffix
    
    # Check for '-ness' suffix
    elif word.endswith('ness') and len(word) > 4:
        base_word = word[:-4]
        morphemes.append(base_word)
        morphemes.append('-ness')
        
    # Check for '-ing' suffix
    elif word.endswith('ing') and len(word) > 3:
        morphemes.append(word[:-3])
        morphemes.append('ing')
        
    # Check for '-ed' suffix
    elif word.endswith('ed') and len(word) > 2:
        morphemes.append(word[:-2])
        morphemes.append('ed')
        
    else:
        morphemes.append(word) # No rules matched
        
    return morphemes

def finite_state_transducer(word):
    """
    A simple FST simulation to handle plurals.
    """
    if word.endswith('s'):
        return word[:-1]  # Remove plural 's'
    return word

# ==========================================
# Step 2: Define Main Analysis Function
# ==========================================
def morphological_analysis(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stop-words removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Text normalization
    normalized_tokens = [word.lower().strip(string.punctuation) for word in filtered_tokens]
    
    # Apply Morphological Processing
    processed_tokens = []
    for word in normalized_tokens:
        # 1. Apply Finite State Transducer
        base_form = finite_state_transducer(word)
        processed_tokens.append(base_form)
        
        # 2. Apply Morphological Parsing (Morpheme extraction)
        morphemes = morphological_parsing(word)
        processed_tokens.extend(morphemes)
        
    return list(set(processed_tokens))  # Return unique tokens

# ==========================================
# Step 3: Execution
# ==========================================
text_sample = "The cats are playing with the ball and they are enjoying No unhappiness reported"
processed_text = morphological_analysis(text_sample)
print(processed_text)