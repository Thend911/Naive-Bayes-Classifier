import numpy as np

class EvaluationMetrics:
    def compute_metrics(TP,FP,TN,FN):
        accuracy = np.divide(np.add(TP,TN),np.add(TP,TN,FP,FN))
        precision= np.divide(TP,np.sum(TP,FP))
        recall = np.divide(TP,np.sum(TP,FN))
        F1=np.multiply(2,np.divide(np.multiply(precision,recall),np.sum(precision,recall)))