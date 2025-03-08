import re
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt_tab')
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
        X,Y = [],[] # Make list that contain msg and label
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Strip leading/trailling whitespace
                # Split only on the first space to separate label
                parts = line.strip().split(maxsplit=1) 
                # If Parts contain label and message (2 part)
                if len(parts) == 2:
                    # Categorize
                    label, text = parts
                    # Process text
                    processed_text = self.preprocess(text)
                    binary_label = 1 if label.lower() == 'spam' else 0  # Convert label to binary
                    X.append(processed_text)
                    Y.append(binary_label)
        return X,Y
    def split_data(self, x, y):
        train_ratio = 0.8

        # Zip x and y together, convert to list, and shuffle
        data = list(zip(x, y))
        random.shuffle(data)

        # Split
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        # Unzip into separate lists
        X_train, Y_train = zip(*train_data)
        X_test, Y_test = zip(*test_data)

        return list(X_train), list(Y_train), list(X_test), list(Y_test)
