import csv
import sys
import numpy as np
from validate import validate

def import_data_and_weights(test_X_file_path,weights_file_path):
    test_X=np.genfromtxt(test_X_file_path,delimiter=',',dtype=np.float64,skip_header=1)
    weights=np.genfromtxt(weights_file_path,delimiter=',',dtype=np.float64)
    return test_X,weights

def predict_target_values(test_X,weights):
    test_X=np.insert(test_X,0,1,axis=1)
    pred_Y=np.dot(test_X,weights.T)
    return pred_Y

def write_to_csv_file(pred_Y,predicted_Y_file_name):
    pred_Y=pred_Y.reshape(len(pred_Y),1)
    with open(predicted_Y_file_name,'w',newline='') as predicted_file:
        file_writer=csv.writer(predicted_file)
        file_writer.writerows(pred_Y)
        predicted_file.close()

def predict(test_X_file_path):
    test_X,weights=import_data_and_weights(test_X_file_path,"WEIGHTS_FILE.csv")
    pred_Y=predict_target_values(test_X,weights)
    write_to_csv_file(pred_Y,"predicted_test_Y_lr.csv")
    
    
if __name__=="__main__":
    test_X_file_path=sys.argv[1] #This help the test_X_file_path as a command line arguments
    predict(test_X_file_path)
    #validate(test_X_file_path,actual_test_Y_file_path="train_Y_lr.csv")
    