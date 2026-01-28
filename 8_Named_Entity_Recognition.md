
# Practical 8: Implement Named Entity Recognition

**Aim:** To implement Named Entity Recognition (NER) to identify and classify entities (such as persons, organizations, and locations) in text using the pre-trained models provided by the Flair library.

## Python Implementation

This script uses the `flair` library to load a pre-trained NER model ('ner') and predict entities in user-provided text.

```python
# ==========================================
# Step 1: Setup and Installation
# ==========================================
pip install flair

from flair.models import SequenceTagger
from flair.data import Sentence

# ==========================================
# Step 2: Define Helper Functions
# ==========================================

def get_user_input():
    """
    Prompts the user to enter a text string for analysis.
    """
    user_input = input("Please enter a text: ")
    return user_input

def named_entity_recognition(text):
    """
    Loads the pre-trained NER model and predicts entities for the given text.
    """
    # Load the pre-trained NER model
    # The 'ner' model identifies 4 classes: PER (Person), LOC (Location), ORG (Organization), MISC
    tagger = SequenceTagger.load("ner")
    
    # Create a Sentence object required by Flair
    sentence = Sentence(text)
    
    # Predict entities
    tagger.predict(sentence)
    return sentence

def print_named_entities(sentence):
    """
    Iterates through the found entities and prints them with their tags.
    """
    print("\nNamed Entities:")
    # .get_spans('ner') retrieves the entities identified by the tagger
    for entity in sentence.get_spans('ner'):
        print(f"- {entity.text} ({entity.tag})")

# ==========================================
# Step 3: Main Execution
# ==========================================
if __name__ == "__main__":
    # Example inputs mentioned in source:
    # 1. Barack Obama was the 44th President of the United States.
    # 2. The Eiffel Tower is one of the most famous landmarks in Paris, France.
    # 3. Apple Inc. announced a new product launch in San Francisco.
    
    # Get text input from user
    text = get_user_input()
    
    # Perform NER
    sentence = named_entity_recognition(text)
    
    # Print results
    print_named_entities(sentence)
```

**Output:**
```text
Please enter a text: The Eiffel Tower is one of the most famous landmarks in Paris, France

Named Entities:
- Eiffel Tower (ORG)
- Paris (LOC)
- France (LOC)
```
*,,,,,*
```