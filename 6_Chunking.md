
# Practical 6: Implement Chunking

**Aim:** To implement chunking (shallow parsing) to extract meaningful phrases such as Noun Phrases and Verb Phrases from sentences using the NLTK library.

## Part 1: Basic Python Chunking Utilities

This section demonstrates how to break raw data (lists or text strings) into smaller, fixed-size segments.

```python
# ==========================================
# 1. Chunking a List
# ==========================================
def chunk_list(data, chunk_size):
    """Yield successive chunk_size-sized chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Example usage:
data =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
chunks = list(chunk_list(data, 3))
print(f"List Chunks: {chunks}")

# ==========================================
# 2. Chunking Text by Character Length
# ==========================================
def chunk_text(text, chunk_size):
    """Break text into chunks of specified character length."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

# Example usage:
text = "This is an example of chunking text into smaller pieces."
text_chunks = list(chunk_text(text, 10))
print(f"Text Chunks: {text_chunks}")
```

**Output:**
```text
List Chunks: [,,]
Text Chunks: ['This is an', ' example o', 'f chunking', ' text into', ' smaller p', 'ieces.']
```
*,*

---

## Part 2: NLP Chunking and Phrase Extraction

This implementation uses NLTK's `RegexpParser` to define grammar rules and extract specific grammatical phrases (Noun Phrases, Verb Phrases, etc.) from a sentence.

```python
import nltk
from nltk import pos_tag, word_tokenize, RegexpParser

# ==========================================
# Step 1: Setup and Downloads
# ==========================================
# Download necessary NLTK resources for tokenization and tagging
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ==========================================
# Step 2: Define Chunking Logic
# ==========================================
def chunk_and_extract_phrases(sentence):
    """
    Tokenizes, tags, and chunks a sentence based on specific grammar rules.
    Extracts Noun Phrases (NP) and Verb Phrases (VP).
    """
    
    # 1. Tokenize the input sentence
    tokens = word_tokenize(sentence)
    
    # 2. Perform Part-of-Speech (POS) tagging
    tagged_tokens = pos_tag(tokens)
    
    # 3. Define grammar for chunking using Regular Expressions
    grammar = """
    NP: {<DT>?<JJ>*<NN.*>}   # Noun Phrase: Optional Determiner + Adjectives + Noun
    VP: {<VB.*><NP|PP>*}     # Verb Phrase: Verb + (Noun Phrase or Prep Phrase)
    AP: {<JJ.*>+}            # Adjective Phrase
    AdvP: {<RB.*>+}          # Adverbial Phrase
    PP: {<IN><NP>}           # Prepositional Phrase
    """
    
    # 4. Create a chunk parser
    chunk_parser = RegexpParser(grammar)
    
    # 5. Parse the tagged tokens to get the chunk tree
    chunked = chunk_parser.parse(tagged_tokens)
    
    # 6. Extract phrases by iterating through the parse tree
    noun_phrases = []
    verb_phrases = []
    
    for subtree in chunked.subtrees():
        # Check if the subtree represents a Noun Phrase
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(word for word, tag in subtree.leaves()))
            
        # Check if the subtree represents a Verb Phrase
        elif subtree.label() == 'VP':
            verb_phrases.append(' '.join(word for word, tag in subtree.leaves()))
            
    # 7. Display Results
    print(f"\nOriginal Sentence: {sentence}")
    
    print("\nNoun Phrases (NP):")
    for np in noun_phrases:
        print(f"- {np}")
        
    print("\nVerb Phrases (VP):")
    for vp in verb_phrases:
        print(f"- {vp}")

# ==========================================
# Step 3: Execution
# ==========================================
if __name__ == "__main__":
    # Example input provided in the practical
    user_input = "I am Kevin. I am a boy. I live in India."
    
    # You can also use input() to get dynamic user input:
    # user_input = input("Please enter a sentence: ")
    
    chunk_and_extract_phrases(user_input)
```

**Output:**
```text
Original Sentence: I am Kevin. I am a boy. I live in India.

Noun Phrases (NP):
- Kevin
- a boy
- India

Verb Phrases (VP):
- am Kevin
- am a boy
- live
```
*,,,,,*
```
