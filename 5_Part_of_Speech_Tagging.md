Based on the **NLP_Zeeshan_Final.pdf** source, here is the Markdown file for **Practical 5**, which focuses on **Part-of-Speech (POS) Tagging**.

```markdown
# Practical 5: Implement Part-of-Speech Tagging

**Aim:** To implement Part-of-Speech (POS) tagging to assign grammatical categories (such as nouns, verbs, adjectives) to words in a text corpus using the NLTK library.

## Python Implementation

This script tokenizes a sample text and applies NLTK's pre-trained perceptron tagger to label each token with its corresponding part of speech.

```python
import nltk

# ==========================================
# Step 1: Setup and Downloads
# ==========================================
# Download necessary NLTK resources for tokenization and tagging
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# ==========================================
# Step 2: Define Helper Functions
# ==========================================

def preprocess(text):
    """
    Tokenizes the input text into individual words.
    """
    tokens = nltk.word_tokenize(text)
    return tokens

def pos_tagging(text):
    """
    Performs Part of Speech tagging on the tokenized text.
    Returns a list of tuples (word, tag).
    """
    # 1. Tokenize the text
    tokens = preprocess(text)
    
    # 2. Apply POS tagging
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens

# ==========================================
# Step 3: Execution
# ==========================================
if __name__ == "__main__":
    # Sample text corpus
    text = """
    Natural language processing (NLP) is a fascinating field of artificial intelligence.
    It allows computers to understand and interpret human language.
    """

    # Perform POS tagging
    tagged_output = pos_tagging(text)

    # Display the tagged tokens
    print("Tagged Tokens:")
    for token, tag in tagged_output:
        print(f"{token}: {tag}")
```

**Output:**
```text
Tagged Tokens:
Natural: JJ
language: NN
processing: NN
(: (
NLP: NNP
): )
is: VBZ
a: DT
fascinating: JJ
field: NN
of: IN
artificial: JJ
intelligence: NN
.: .
It: PRP
allows: VBZ
computers: NNS
to: TO
understand: VB
and: CC
interpret: VB
human: JJ
language: NN
.: .
```

### Key POS Tag Explanations:
*   **NN:** Noun, singular or mass
*   **JJ:** Adjective
*   **VBZ:** Verb, 3rd person singular present
*   **DT:** Determiner
*   **NNP:** Proper noun, singular
*   **PRP:** Personal pronoun
```