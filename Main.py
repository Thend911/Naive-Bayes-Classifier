import logging
from DataLoader import DataLoader
from NaiveBayes import Naive_Bayes
from Evaluationmetrics import EvaluationMetrics

file_path = "SMSSPamCollection.txt"
data = DataLoader(file_path)
cleaned_data = data.load_data()
train,test=data.split_data(cleaned_data)

nb = Naive_Bayes()
p_y, p_ny, p_xy, p_xny=nb.train(train)
predicted_train = nb.prediction(p_y, p_ny, p_xy, p_xny,test)
p_y_t, p_ny_t, p_xy_t, p_xny_t=nb.train(predicted_train)

em = EvaluationMetrics()
train_tp, train_tn,train_fp,train_fn,train_accuracy,train_f1=em.compute_metrics(p_y, p_ny, p_xy, p_xny)
test_tp, test_tn,test_fp,test_fn,test_accuracy,test_f1=em.compute_metrics(p_y_t, p_ny_t, p_xy_t)

logging.basicConfig(filename='result.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Number of training samples: {len(train)}")
logging.info(f"Number of testing samples: {len(predicted_train)}")
logging.info(f"Training Accuracy: {train_accuracy}")
logging.info(f"Testing Accuracy: {test_accuracy}")
logging.info(f"Training F1 Score: {train_f1}")
logging.info(f"Testing F1 Score: {test_f1}")
logging.info(f"Training Confusion Matrix (TP, FN, FP, TN): {train_tp}, {train_fn}, {train_fp}, {train_tn}")
logging.info(f"Testing Confusion Matrix (TP, FN, FP, TN): {test_tp}, {test_fn}, {test_fp}, {test_tn}")