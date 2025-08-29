# Normalization :
# 𝑋(𝑛𝑜𝑟𝑚) = X−μ/σ

# Mean = 0
# Standard deviation = 1
# This makes features comparable in scale and keeps gradients stable.

import numpy as np
import matplotlib.pyplot as plt

X = np.array([650, 800, 1000, 1200, 1400, 1600, 1800])  # feature: area
y = np.array([70000, 85000, 100000, 120000, 140000, 160000, 180000])  # target: price

X = (X-np.mean(X))/np.std(X)

m = 0.0
c=0.0
learning_rate = 0.01
rounds = 1000
n = len(X)

def calculate_MC():
    global m,c
    for i in range(1000):
        dm = (-2/n)*np.sum(X*(y-(m*X+c)))
        dc = (-2/n)*np.sum(y-(m*X+c))
        m = m-(learning_rate*dm)
        c = c-(learning_rate*dc)

def Predicted_Area(area):
    area_norm = (area - np.mean([650,800,1000,1200,1400,1600,1800])) / np.std([650,800,1000,1200,1400,1600,1800])
    return m*area_norm+c

def Plotting():
    plt.title("RESULT OF REGRESSION LAYER")
    plt.scatter(X,y,color = 'blue',label="Original Output")
    plt.plot(X,m*X+c,"r-",label="Predicted Output")
    plt.scatter(user_input,user_output,color="green")
    plt.legend()
    plt.show()

user_input =[] 
user_output=[]
def Main():
    global user_input
    global user_output
    choice = input("Enter y : Predict Price \nEnter n : Exit!!!\nChoice : ")

    if(choice=='y'):
        area = float(input("Enter The Area : "))
        price = Predicted_Area(area)
        print(price)
        user_input.append((area- np.mean([650,800,1000,1200,1400,1600,1800])) / np.std([650,800,1000,1200,1400,1600,1800]))
        user_output.append(price)
        Main()


calculate_MC()
Main()
Plotting()