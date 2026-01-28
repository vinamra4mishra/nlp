Here is the complete code for **Practical 1: Text Pre-processing** formatted as a Markdown file with explanatory comments, based on the provided source,,,,,,.

```markdown
# Practical 1: Text Pre-processing

**Aim:** To perform pre-processing methods for text data including tokenization, stop-word removal, and normalization using the NLTK library.

**Input Data:** `SMSSpamCollection`
**Output Data:** `processed_output.txt`

## Python Implementation

```python
# ==========================================
# Step 1: Setup and Installation
# ==========================================
# Ensure necessary libraries are installed
# !pip install nltk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK datasets for tokenization and stop-words
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ==========================================
# Step 2: Define Helper Functions
# ==========================================

def tokenize(text):
    """
    Tokenizes the input text into individual words.
    """
    return word_tokenize(text)

def remove_stopwords(tokens):
    """
    Removes common English stop words (e.g., 'is', 'the', 'in') from the list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def normalize(text):
    """
    Normalizes the text by converting it to lowercase and removing punctuation.
    """
    text = text.lower()  # Convert to lowercase
    # Remove punctuation using a translation table
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ==========================================
# Step 3: Create the Processing Pipeline
# ==========================================

def preprocess_text(text):
    """
    Combines all text preprocessing steps into a single pipeline:
    1. Normalize (Lowercase + Remove Punctuation)
    2. Tokenize (Split into words)
    3. Filter (Remove Stop-words)
    4. Rejoin (Return as a single string)
    """
    normalized_text = normalize(text) 
    tokens = tokenize(normalized_text) 
    filtered_tokens = remove_stopwords(tokens) 
    return ' '.join(filtered_tokens)

# ==========================================
# Step 4: Process the Data File
# ==========================================

# Define file paths
input_file_path = '/content/SMSSpamCollection'
output_file_path = 'processed_output.txt'

# Read the raw input file
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Apply the preprocessing pipeline to each line
processed_lines = [preprocess_text(line.strip()) for line in lines]

# Save the processed results to a new file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for line in processed_lines:
        file.write(line + '\n')

print(f"Text preprocessing completed and saved to '{output_file_path}'.")

# ==========================================
# Step 5: Verification (View Results)
# ==========================================

# Print the first 5 lines of the ORIGINAL input file for comparison
print("\nProvided Input (First 5 Lines):")
with open(input_file_path, 'r', encoding='utf-8') as file:
    for _ in range(5):
        line = file.readline()
        if line:
            print(line.strip())
        else:
            break

# Print the first 5 lines of the PROCESSED output file
print("\nProcessed Output (First 5 Lines):")
with open(output_file_path, 'r', encoding='utf-8') as file:
    for _ in range(5):
        line = file.readline()
        if line:
            print(line.strip())
        else:
            break
