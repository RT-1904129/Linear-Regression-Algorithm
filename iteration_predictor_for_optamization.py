import numpy as np
import matplotlib.pyplot as plt

Threshold_value_of_difference_of_cost_values=0.000001

def Import_data():
    X=np.genfromtxt("train_X_lr.csv",delimiter=',',dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_lr.csv",delimiter=',',dtype=np.float64)
    return X,Y

def Compute_gradient_of_cost_function(X,Y,W):
    Y_pred=X.dot(W.T)
    difference=Y_pred-Y
    dw=(np.dot(difference.T,X))/X.shape[0]
    return dw

def Compute_cost(X,Y,W):
    Y_pred=np.dot(X,W.T)
    Mean_Squared_error=np.sum(np.square(Y_pred-Y))
    cost_value=Mean_Squared_error/(2*(X.shape[0]))
    return cost_value

def Optimize_weights_using_gradient_descent(X,Y,W,learning_rate):
    i=1;
    prev_cost_value=0
    while True:
        dw=Compute_gradient_of_cost_function(X,Y,W)
        W=W-(learning_rate*dw)
        cost_value=Compute_cost(X,Y,W)
        if i%10000==0:
            print("current i value",i,"cost value--",cost_value)
        if abs(cost_value-prev_cost_value)<(Threshold_value_of_difference_of_cost_values):
            print("final no of iteration",i)
            break
        prev_cost_value=cost_value
        i+=1
    return W

def train_model(X,Y):
    X=np.insert(X,0,1,axis=1)
    Y=Y.reshape(X.shape[0],1)
    W=np.zeros((1,X.shape[1]))
    W=Optimize_weights_using_gradient_descent(X,Y,W,0.0001)
    return W

if __name__=="__main__":
    X,Y=Import_data()
    weights=train_model(X,Y)
