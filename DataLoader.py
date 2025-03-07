import re
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class DataLoader:
    def __init__(self,file_path):
        self.file_path = file_path
        # Define method before using in class
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        # Convert text to lowercase
        text = text.lower()
        # Remove special characters and punctuation
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Tokenize words
        words = word_tokenize(text)
        # Apply stemming and remove stopwords
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        # Join words again to produce sentence
        return ' '.join(words)

    def load_data(self):
        data_set = set() # Make unordered set that contain msg and label
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Strip leading/trailling whitespace
                # Split only on the first space to separate label
                parts = line.strip().split(' ', 1)  
                # If Parts contain label and message (2 part)
                if len(parts) == 2:
                    # Categorize
                    label, text = parts
                    # Process text
                    processed_text = self.preprocess(text)
                    binary_label = 1 if label.lower() == 'spam' else 0  # Convert label to binary
                    data_set.add((processed_text, binary_label))
        return data_set
    def split_data(self, data):
        train_ratio=0.8
        random.shuffle(data)  # Shuffle dataset
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]  # First 80%
        test_data = data[split_idx:]   # Remaining 20%
        return train_data, test_data