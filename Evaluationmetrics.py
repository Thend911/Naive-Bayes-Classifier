import numpy as np

class EvaluationMetrics:
    def compute_metrics(p_y, p_ny, p_xy, p_xny):
        TP = p_y*p_xy           #         /P(xy)            #True Pos
        FN = p_y *(1-p_xy)      #    P(y)/\1-P(xy)=P(nxy)   #False Neg
                                #       ~
        FP = p_ny * p_xny       #   P(ny)\/P(xny)           #False Pos
        TN = p_ny * (1-p_xny)   #         \1-P(xny)=P(nxny) #True Neg
        accuracy = np.divide(np.add(TP,TN),np.add(TP,TN,FP,FN))
        precision= np.divide(TP,np.sum(TP,FP))
        recall = np.divide(TP,np.sum(TP,FN))
        F1=np.multiply(2,np.divide(np.multiply(precision,recall),np.sum(precision,recall)))
        return TP,TN,FP,FN,accuracy,F1