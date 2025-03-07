import numpy as np
import collections
from collections import Counter
from collections import defaultdict


class Naive_Bayes:
    def train(train_data):
        total = ham = spam = W_ham = W_spam = 0
        wham = wspam = defaultdict(int)
        p_wspam = p_wham =defaultdict(int)
        # Scan for label and count word in data
        for words,label in train_data:
            token = words.split()
            if label:
                spam+=1
                for word in token:
                    wspam[word]+=1
                    W_spam+=1
            else:
                ham+=1
                for word in token:
                    wham[word]+=1
                    W_ham+=1
            total+=1
        
        # P(y/x) in which y is spam and ~y is ham, x is text
        p_spam =np.log(spam/total) # log P(y)
        p_ham = np.log(ham/total) # log P(~y)

        # P(x/y)
        # P(x/~y)
        # Laplace smoothing to avoid log(0), because sometimes the testing data may have word that training does not have, smoothing to avoid 0 probability 
        smoothing_factor = 1  # Additive smoothing to handle unseen words
        p_wspam = {word: np.log((count + smoothing_factor) / (W_spam + smoothing_factor * len(wspam))) for word, count in wspam.items()} #log P(x/y)
        p_wham = {word: np.log((count + smoothing_factor) / (W_ham + smoothing_factor * len(wham))) for word, count in wham.items()} # log P(x/~y)
        return p_spam,p_ham,p_wspam,p_wham
    

    def prediction(p_spam,p_ham,p_wspam,p_wham):
        
        return