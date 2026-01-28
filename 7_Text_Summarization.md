
# Practical 7: Implement Text Summarization

**Aim:** To implement text summarization techniques (Extractive and Abstractive) using Python libraries.

## Part 1: Extractive Summarization (using Sumy)

Extractive summarization selects the most important sentences from the original text to create a summary. This implementation uses the Latent Semantic Analysis (LSA) summarizer from the `sumy` library.

```python
# ==========================================
# Step 1: Setup and Installation
# ==========================================
!pip install sumy
import nltk
# Ensure NLTK resources are available
nltk.download('punkt')

# Import necessary libraries
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# ==========================================
# Step 2: Define Helper Functions
# ==========================================

def get_text_input():
    """Function to get long text input from the user via console."""
    print("Please enter the text you want to summarize (end with a blank line):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

def summarize_text(text, sentence_count=3):
    """
    Function to summarize the text using Sumy's LSA Summarizer.
    
    Args:
        text (str): Input text to summarize.
        sentence_count (int): Number of sentences to include in the summary.
    """
    try:
        # Parse the text using the English tokenizer
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        
        # Generate summary
        summary = summarizer(parser.document, sentence_count)
        
        # Join the summarized sentences into a single string
        return ' '.join(str(sentence) for sentence in summary)
    except Exception as e:
        return f"Error in summarization: {str(e)}"

# ==========================================
# Step 3: Main Execution
# ==========================================
def main():
    """Main function to run the text summarization."""
    # Get input text
    long_text = get_text_input()
    
    # Create a output file and write original and summarized text to it
    output_filename = 'extractivesummarized.txt'
    
    with open(output_filename, 'w') as file:
        file.write("Original Text:\n")
        file.write(long_text + "\n\n")
        
        # Perform summarization
        summary = summarize_text(long_text)
        
        file.write("Summarized Text:\n")
        file.write(summary)
        
    print(f"The original and summarized texts have been written to {output_filename}.")

if __name__ == "__main__":
    main()
```
*,,,*

---

## Part 2: Abstractive Summarization (using Hugging Face Transformers)

Abstractive summarization generates new sentences to capture the meaning of the original text, similar to how a human would summarize. This implementation uses the `facebook/bart-large-cnn` model via the `transformers` pipeline.

```python
# ==========================================
# Step 1: Setup and Installation
# ==========================================
# !pip install transformers torch
import nltk
from transformers import pipeline

# Ensure NLTK resources are available
nltk.download('punkt')

# ==========================================
# Step 2: Define Summarization Logic
# ==========================================

def get_text_input():
    """Function to get long text input from the user."""
    print("Please enter the text you want to summarize (end with a blank line):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

def summarize_text(text):
    """
    Function to summarize the text using Hugging Face Transformers.
    Uses the 'facebook/bart-large-cnn' model.
    """
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Generate summary with specific constraints
    # max_length: Maximum length of the summary
    # min_length: Minimum length of the summary
    # do_sample: False disables sampling (greedy decoding)
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    
    return summary['summary_text']

# ==========================================
# Step 3: Main Execution
# ==========================================
def main():
    """Main function to run the text summarization."""
    # Get input text
    long_text = get_text_input()
    
    # Define output file name
    output_filename = 'abstractivesummarized.txt'
    
    # Write original and summarized text to file
    with open(output_filename, 'w') as file:
        file.write("Original Text:\n")
        file.write(long_text + "\n\n")
        
        # Perform summarization
        summary = summarize_text(long_text)
        
        file.write("Summarized Text:\n")
        file.write(summary)
        
    print(f"The original and summarized texts have been written to {output_filename}.")

if __name__ == "__main__":
    main()
```
*,,*
```
