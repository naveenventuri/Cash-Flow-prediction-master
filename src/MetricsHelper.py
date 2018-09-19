from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,classification_report, hamming_loss
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def feature_importance(feature_importances_, x_train):
    
    feature_importances = pd.DataFrame(feature_importances_,
                                       range(x_train.shape[1]),
                                       columns=['importance']).sort_values('importance',ascending=False)
                                       
    for row in feature_importances.itertuples():
        print(row)
        
def gen_metrics(y_predict, y_test, verbose=False):
    
    y_predict = np.array(y_predict)
    y_test = np.array(y_test)
    
    total_dist = 0
    total_sqr_dist = 0
    for i in range(len(y_predict)):
        dist = (y_predict[i] - y_test[i])
        if verbose:
            print("Predicted, Test, Distance :",y_test[i] , y_predict[i], dist )
        
        total_dist += dist
        total_sqr_dist += dist**2
        
    total_dist/=len(y_test) 
    total_sqr_dist/=len(y_test) 
    print("Mean distance, RMSE :", total_dist, math.sqrt(total_sqr_dist))
    
def compare_metrics(y_test, y_predict1, y_predict2):
    
    y_predict1 = np.array(y_predict1)
    y_predict2 = np.array(y_predict2)
    y_test = np.array(y_test)
    
    total_dist1 = 0
    total_sqr_dist1 = 0
    total_dist2 = 0
    total_sqr_dist2 = 0
    for i in range(len(y_test)):
        dist1 = (y_predict1[i] - y_test[i])
        dist2 = (y_predict2[i] - y_test[i])
        
        print("Test, Predicted1, Distance , Predicted2, Distance:",y_test[i] , y_predict1[i], dist1, y_predict2[i], dist2 )
        
        total_dist1 += dist1
        total_sqr_dist1 += dist1**2
        total_dist2 += dist2
        total_sqr_dist2 += dist2**2
        
    total_dist1/=len(y_test) 
    total_sqr_dist1/=len(y_test) 
    total_dist2/=len(y_test) 
    total_sqr_dist2/=len(y_test) 
    print("Model 1: Mean distance, RMSE :", total_dist1, math.sqrt(total_sqr_dist1))
    print("Model 1: Mean distance, RMSE :", total_dist2, math.sqrt(total_sqr_dist2))
    
    print("--------------------------------------------------")    