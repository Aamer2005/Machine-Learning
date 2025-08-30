import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from flask import Flask, render_template, request

app = Flask(__name__)

df = pd.read_csv('score.csv')
print(df.head(10))

X = df['Hours'].values
Y = df['Score'].values

# X = X-X.mean() / 

m = 0
c=0
iterations = 1000
learning_rate = 0.01
n = len(Y)
cost_history = []

for _ in range(iterations):
    error = Y-(m*X+c)
    MSE = (1/n)*np.sum(error**2)
    cost_history.append(MSE)

    dm = (-2/n)*np.sum(X*(error))
    dc = (-2/n)*np.sum(error)

    m = m-(learning_rate*dm)
    c = c-(learning_rate*dc)

training_set_x = []
training_set_y = []

def Main():
    global m
    global c 

    choice = input("Enter y : Predict Score \nEnter n : Exit!!!\nChoice : ")

    if(choice=='y'):
        X_user = float(input("Enter Study Hours : "))
        Y_cap = m*X_user + c
        Y_cap = min(Y_cap, 100)   # maximum 100
        Y_cap = max(Y_cap, 0)  
        print('Y : ',Y_cap)
        training_set_x.append(X_user)
        training_set_y.append(Y_cap)
        Main()
    else:
        plt.title("Linear Regression")
        plt.scatter(X,Y,color='blue',label='Training Dataset')
        plt.scatter(np.array(training_set_x),np.array(training_set_y),color = 'green',label='training set')
        plt.plot(X,m*X+c,'r-',label='Regression Line')
        plt.xlabel("Hours Of Study")
        plt.ylabel('Score')
        plt.legend()
        plt.show()

# plt.title("Linear Regression")
# plt.scatter(X,Y,color='blue',label='Training Dataset')
# plt.plot(X,m*X+c,'r-',label='Regression Line')
# plt.xlabel("Hours Of Study")
# plt.ylabel('Score')
# plt.legend()
# plt.show()

Main()

#Plotting MSE

plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.legend()
plt.show()
