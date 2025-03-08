import logging
from DataLoader import DataLoader
from NaiveBayes import Naive_Bayes
from Evaluationmetrics import EvaluationMetrics


file_path = "SMSSPamCollection.txt"
data = DataLoader(file_path)
x,y=[],[]
x,y = data.load_data()
train_text, train_label,test_text,test_label=data.split_data(x,y)

nb = Naive_Bayes()
p_y, p_ny, p_xy, p_xny=nb.train(train_text, train_label)
upd_train_text,upd_train_label=nb.prediction(p_y, p_ny, p_xy, p_xny,train_text)
upd_test_text,upd_test_label=nb.prediction(p_y, p_ny, p_xy, p_xny,test_text)

#TEST AREA######
#########################
#print(p_y, p_ny)
#########################
#(test_label,"\n",upd_test_label)
em = EvaluationMetrics()
train_tp,   train_tn,   train_fp,   train_fn,   train_accuracy, train_f1    =em.compute_metrics(train_label,upd_train_label)
test_tp,    test_tn,    test_fp,    test_fn,    test_accuracy,  test_f1     =em.compute_metrics(test_label,upd_test_label)

logging.basicConfig(filename='result.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Number of training samples: {len(train_text)}")
logging.info(f"Number of testing samples: {len(test_text)}")
logging.info(f"Training Accuracy: {train_accuracy}")
logging.info(f"Testing Accuracy: {test_accuracy}")
logging.info(f"Training F1 Score: {train_f1}")
logging.info(f"Testing F1 Score: {test_f1}")
logging.info(f"Training Confusion Matrix (TP, FN, FP, TN): {train_tp}, {train_fn}, {train_fp}, {train_tn}")
logging.info(f"Testing Confusion Matrix (TP, FN, FP, TN): {test_tp}, {test_fn}, {test_fp}, {test_tn}")