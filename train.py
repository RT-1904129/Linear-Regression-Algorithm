import csv
import numpy as np
import matplotlib.pyplot as plt
list_cost_value=[]
list_cost_index=[]
learning_rate=0.0001
no_iterations=98525498

# This function read the training csv file
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

def Optimize_weights_using_gradient_descent(X,Y,W,no_iterations,learning_rate):
    for i in range(1,no_iterations):
        dw=Compute_gradient_of_cost_function(X,Y,W)
        W=W-(learning_rate*dw)
        cost_value=Compute_cost(X,Y,W)
        list_cost_index.append(i)
        list_cost_value.append(cost_value)
    return W

def train_model(X,Y):
    X=np.insert(X,0,1,axis=1)
    Y=Y.reshape(X.shape[0],1)
    W=np.zeros((1,X.shape[1]))
    W=Optimize_weights_using_gradient_descent(X,Y,W,no_iterations,learning_rate)
    return W

# Here we use csv module to write W values in newly created csv file
def save_model(weights,weights_file_name):
    with open(weights_file_name,'w',newline='') as weight_file:# we use newline='' because i want to remove blank row during writing 
        file_writer=csv.writer(weight_file,delimiter=",") #csv.writer(weight_file,delimiter=in what way you want sepration)
        file_writer.writerows(weights) # for writing arrays row in rows of created csv file
        weight_file.close()
        
if __name__=="__main__":
    X,Y=Import_data()
    weights=train_model(X,Y)
    plt.plot(list_cost_index,list_cost_value)
    save_model(weights,"WEIGHTS_FILE.csv")