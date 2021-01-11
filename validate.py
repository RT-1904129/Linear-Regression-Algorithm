import csv
from os import path
import numpy as np


def checking_file_exits(predicted_test_Y_file_path):
    if not path.exists(predicted_test_Y_file_path):
        raise Exception("Couldn't find '"+predicted_test_Y_file_path+"' file")


def checking_format(test_X_file_path,predicted_test_Y_file_path):
    with open(predicted_test_Y_file_path,'r') as file:
        file_reader=csv.reader(file)
        pred_Y=np.array(list(file_reader))
        file.close()
        test_X=np.genfromtxt(test_X_file_path,delimiter=',',dtype=np.float64,skip_header=1)
    if pred_Y.shape!=(test_X.shape[0],1):
        raise Exception("Output format is not proper")

def checking_mse(actual_test_Y_file_path,predict_test_Y_file_path):
    pred_Y=np.genfromtxt(predict_test_Y_file_path,delimiter=',',dtype=np.float64)
    actual_Y=np.genfromtxt(actual_test_Y_file_path,delimiter=',',dtype=np.float64)
    from sklearn.metrics import mean_squared_error
    mse= mean_squared_error(actual_Y,pred_Y)
    return mse
        
def validate(test_X_file_path,actual_test_Y_file_path):
    predicted_test_Y_file_path="predicted_test_Y_lr.csv"
    checking_file_exits(predicted_test_Y_file_path)
    checking_format(test_X_file_path,predicted_test_Y_file_path)
    print(checking_mse(actual_test_Y_file_path,predicted_test_Y_file_path))
    
