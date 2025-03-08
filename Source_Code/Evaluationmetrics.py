import numpy as np

class EvaluationMetrics:
    def compute_metrics(self,actual_labels,predicted_labels):
        #         / P(x|y)              #True Pos
        #    P(y)/\ P(nx|y)=1-P(x|y)    #False Neg    
        #       ~
        #   P(ny)\/ P(x|ny)             #False Pos
        #         \ P(nx|ny)=1-P(x|ny)  #True 
        TP=TN=FP=FN=0
        for label_a, label_p in zip(actual_labels, predicted_labels):
            if label_a == 1 and label_p == 1:
                TP += 1
            elif label_a == 1 and label_p == 0:
                FN += 1
            elif label_a == 0 and label_p == 1:
                FP += 1
            elif label_a == 0 and label_p == 0:
                TN += 1
        
        accuracy = (TP + TN)/(TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0 
        precision = TP/(TP + FP) if (TP + FP) > 0 else 0
        recall = TP/(TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * (precision * recall)/(precision + recall) if (precision + recall) > 0 else 0
        return TP,TN,FP,FN,accuracy,F1