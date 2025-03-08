import numpy as np
import collections
from collections import Counter
from collections import defaultdict


class Naive_Bayes:
    def train(self,train_x, train_y):
        total = 0
        ham=0
        spam=0
        W_ham=0
        W_spam=0
        wham = defaultdict(int)
        wspam = defaultdict(int)

        # Scan for label and count word in data
        data=list(zip(train_x,train_y))
        for words,label in data:
            if label==1:
                spam+=1
                for word in words:
                    wspam[word]+=1
                    W_spam+=1
            else:
                ham+=1
                for word in words:
                    wham[word]+=1
                    W_ham+=1
            total+=1

        # Laplace smoothing to avoid log(0), because sometimes the testing data may have word that training does not have, smoothing to avoid 0 probability 
        smoothing_factor = 1  # Additive smoothing to handle unseen words

        # P(y/x) in which y is spam and ~y is ham, x is text
        p_spam =np.log((spam+smoothing_factor)/(total+2*smoothing_factor)) # log P(y)
        p_ham = np.log((ham+smoothing_factor)/(total+2*smoothing_factor)) # log P(~y)

        # P(x/y)
        # P(x/~y)
        total_unique_words = len(set(wspam.keys()).union(set(wham.keys())))
        p_wspam = {word: np.log((count + smoothing_factor) / (W_spam + smoothing_factor * total_unique_words)) for word, count in wspam.items()} #log P(x/y)
        p_wham = {word: np.log((count + smoothing_factor) / (W_ham + smoothing_factor * total_unique_words)) for word, count in wham.items()} # log P(x/~y)

                #P(y)   #P(~y)  #P(x/y)     #P(x/~y)  
        return  p_spam, p_ham,  p_wspam,    p_wham
    

    def prediction(self,p_spam, p_ham, p_wspam, p_wham, x):
        updated_y=[]
        for words in x:
            ham = p_ham
            spam = p_spam
            for word in words:
                if word in p_wspam and word in p_wham:
                    spam += p_wspam[word]
                    ham += p_wham[word]
            if ham > spam:
                updated_y.append(0)
            else:
                updated_y.append(1)
        
        return x,updated_y  # return data after prediction