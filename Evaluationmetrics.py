import numpy as np

class EvaluationMetrics:
    def compute_metrics(p_y, p_ny, p_xy, p_xny):
        TP = p_y*p_xy           #         / P(x|y)               #True Pos
        FN = p_y *(1-p_xy)      #    P(y)/\ P(nx|y)=1-P(x|y)     #False Neg
                                #       ~
        FP = p_ny * p_xny       #   P(ny)\/ P(x|ny)              #False Pos
        TN = p_ny * (1-p_xny)   #         \ P(nx|ny)=1-P(x|ny)   #True Neg
        accuracy = np.divide(np.add(TP,TN),np.add(TP,TN,FP,FN))
        precision= np.divide(TP,np.sum(TP,FP))
        recall = np.divide(TP,np.sum(TP,FN))
        F1=np.multiply(2,np.divide(np.multiply(precision,recall),np.sum(precision,recall)))
        return TP,TN,FP,FN,accuracy,F1