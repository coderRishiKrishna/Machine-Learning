import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import normalize


def load_data(path):
    data = pd.read_csv(path,header = 0)
    return data

def prepare_data(data):
    x = data.loc[:,data.columns !="target"].values
    y = data.iloc[:,-1]
    numpy_y = y.values.reshape(y.shape[0],1)
    return x,numpy_y

def normalize_data(x):
    mins = np.min(x, axis = 0) 
    maxs = np.max(x, axis = 0) 
    max_min_range = maxs - mins 
    noramlised_x = 1 - ((maxs - x)/max_min_range)
     
    return noramlised_x

def plot_data(theta,x,y):
    failed = x[np.where(y==0),:][0]
    passed = x[np.where(y==1),:][0]
    plt.figure()
    plt.scatter(passed[:,1],passed[:,2],s =30,color = "r",marker = "o", label = "passed")
    plt.scatter(failed[:,1],failed[:,2],s = 30,color = "b" ,marker = "x", label = "failed")
    x1 = np.arange(0, 1, 0.1) 
    x2 = -(theta[0] + theta[1]*x1)/theta[2] 
    plt.plot(x1, x2, c='k', label='reg line') 
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    plt.legend() 
    plt.show()  

def sigmoid(x):
    return 1/(1+np.exp(-x))

def linear_output(x,theta):
    return np.dot(x,theta)

def output(x,theta):
    lin_output = linear_output(x,theta)
    return sigmoid(lin_output)

def cost(x,y,theta):
    m = len(x)
    result = output(x,theta)
    total_error = -(1/m)*(np.sum((y*np.log(result))+((1-y)*np.log(1-result))))
    return total_error
    
def Gradient_descent(x,y,theta,learning_rate,max_iterations,min_value):
    cost_per_iteration = [cost(x,y,theta)]
    curr_cost = cost(x,y,theta)
    iterations = 1
    cost_difference = 1
    m = len(x)
    
    while(cost_difference > min_value and iterations <= max_iterations):
        predicted = output(x,theta)
        difference = predicted - y
        delta_theta = np.dot(x.T,difference)
        theta = theta - ((learning_rate/m)*delta_theta)
        new_cost = cost(x,y,theta)
        cost_difference = curr_cost - new_cost
        curr_cost = new_cost
        iterations+=1
        cost_per_iteration.append(new_cost)
        

    return theta,cost_per_iteration,iterations


class Logisticregression():
    def __init__(self,max_iterations,min_cost_difference,learning_rate):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.errors = []
        self.min_cost_difference = min_cost_difference
        self.iterations = 0
        

    def fit(self,x,y):
        self.x = x
        self.y = y
        self.theta = np.zeros((x.shape[1],1))
        theta,errors,iterations = Gradient_descent(self.x,self.y,self.theta,self.learning_rate,self.max_iterations,self.min_cost_difference)
        self.theta = theta
        self.errors = errors
        self.iterations = iterations

    def plot_error(self):
        plt.figure()
        plt.scatter([i for i in range(self.iterations)],self.errors)
        plt.xlabel("No. of Iterations")
        plt.ylabel(" Training Error")
        plt.show()

    def plot_decision_boundary(self,x,y):
        plot_data(self.theta,x,y)
    
    def predict(self,x):
        predicted_value = output(x,self.theta)
        predicted_value = np.where(predicted_value >=0.5, 1, 0)
        predicted_value = predicted_value.reshape(1,x.shape[0])
        return predicted_value

    def accuracy(self,predicted_value,original_value):
        if len(predicted_value[0])!=len(original_value[0]):
            print("dimensions of predicted target data and original target data does not match")
            return
        c = 0
        for x,y in zip(predicted_value[0],original_value[0]):
            if x==y:
                c+=1
        return (c/len(predicted_value[0]))*100



    

if __name__ == "__main__":
 #----------- - loading the data ------------------------------
    data = load_data("data.csv")
#----------- - the last column in data is labelled as "target" and is used to access th target values-------------------
    x,y = prepare_data(data)
    x = normalize_data(x)
    x = np.c_[np.ones((x.shape[0], 1)), x]

#- ----------------------- specifying the classifiers details- ----------------------
    max_iterations = 10000
    minimum_cost_difference = 0.00001
    learning_rate = 0.1
    classifier = Logisticregression(max_iterations,minimum_cost_difference,learning_rate)
    classifier.fit(x,y)
    classifier.plot_error()
    classifier.plot_decision_boundary(x,y)

    print("Value of regression coefficients : ",classifier.theta.reshape(1,classifier.theta.shape[0]))
    print("Number of iterations taken = ", classifier.iterations)


    predicted_values = classifier.predict(x)
#-------- reshaping the target value array from shape(100,1) to (1,100) to match with the predicted valuees shape ------------------
    target_values = y.reshape(1,y.shape[0])

    score = classifier.accuracy(predicted_values,target_values)
    print( "Correctly identified points percentage  = ",score,"%")

    





