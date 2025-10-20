import numpy as np 
import matplotlib.pyplot as plt 

input = np.array([1,2,3,4,5,6,7,8,9,10])
output = np.array([1,1.99,3.01,4.3,5,5.8,7.12,7.99,9.1,10])

def Gradient(m,c):

    n = len(output)
    m_dash = (-2/n)*np.sum(input*(output-(m*input+c)))
    c_dash = (-2/n)*np.sum(output-(m*input+c))

    mue = 0.01
    m = m-(mue*m_dash)
    c = c-(mue*c_dash)
    return m,c

def calculate_predicted_output(m,c,input):
    predicted_output = []
    for x in input:
        predicted_output.append((m*x)+c)
    
    return predicted_output

def Plotting(input , predicted_output):
    plt.title("Linear Regression")
    plt.scatter(input , output , color = "blue")
    plt.plot(input , predicted_output , "r-")
    plt.show()

#Improves Linear Reqgression
m,c = Gradient(0,0)
predicted_output = calculate_predicted_output(m,c,input)
Plotting(input , predicted_output)
print("Y=",m,"X+",c)

#Improve Linear Regression
m,c = Gradient(m,c)
predicted_output = calculate_predicted_output(m,c,input)
Plotting(input,predicted_output)
print("Y=",m,"X+",c)

#Improve Linear Regression
m,c = Gradient(m,c)
predicted_output = calculate_predicted_output(m,c,input)
Plotting(input,predicted_output)
print("Y=",m,"X+",c)

#Improve Linear Regression
m,c = Gradient(m,c)
predicted_output = calculate_predicted_output(m,c,input)
Plotting(input,predicted_output)
print("Y=",m,"X+",c)

#Improve Linear Regression
m,c = Gradient(m,c)
predicted_output = calculate_predicted_output(m,c,input)
Plotting(input,predicted_output)
print("Y=",m,"X+",c)

#Improve Linear Regression
m,c = Gradient(m,c)
predicted_output = calculate_predicted_output(m,c,input)
Plotting(input,predicted_output)
print("Y=",m,"X+",c)

#Improve Linear Regression
m,c = Gradient(m,c)
predicted_output = calculate_predicted_output(m,c,input)
Plotting(input,predicted_output)
print("Y=",m,"X+",c)

#Original output
Plotting(input , output)
