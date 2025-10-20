import numpy as np
import matplotlib.pyplot as plt

#data
input = [ 1 , 2 , 3 , 4  ,5,  6,  7, 8, 9, 10, 11, 12]
output = [ 5, 6, 10, 10, 13 ,14, 16, 19, 21 ,22 ,25, 27]

X = input[:8]
Y = output[:8]

# X = np.array(X)
# Y = np.array(Y)


#testing set a
X_test = input[8:]
Y_test = output[8:]


#Gradient Descent

def GradientDescent(X,Y,m,c):
  # Initialize parameters

  alpha = 0.01  # learning rate

  n = len(X)
  Y_pred = []

  for i in X:
      # Compute predictions
      Y_pred.append( m*i + c)
      
  Y_pred = np.array(Y_pred)

  # Compute gradients
  m_grad = (-2/n) * np.sum(X * (Y - Y_pred))
  c_grad = (-2/n) * np.sum(Y - Y_pred)
      
  # Update parameters
  m = m - alpha * m_grad
  c = c - alpha * c_grad

  print("y_cap = ",m,"X + ",c)
  return m,c

def plotting(m,c):

  y_cap = []

  #Predicted Y_cap values
  for x in X:
    y_cap.append(m*x+c)

  #list to array
  y_cap = np.array(y_cap)

  #Plot LR equation or training data
  plt.title('Changing c and m Or Increasing')
  plt.scatter(X, Y, color='b', label='Actual Data')
  plt.plot(X,y_cap,"r-")
  plt.legend()
  plt.show()

  #y_cap Of testing data
  y_cap_test = []
  for x in X_test:
    y_cap_test.append(m*x+c)

  y_cap_test = np.array(y_cap_test)


  #plot testing data with predicted output
  plt.title('Testing Data plot ')
  plt.scatter(X_test,Y_test,color='b',label='Actual Data')
  plt.plot(X_test,y_cap_test,"y-")
  plt.legend()
  plt.show()



m,c = GradientDescent(X,Y,0,0)
plotting(m,c)
m,c = GradientDescent(X,Y,m,c)
plotting(m,c)
m,c = GradientDescent(X,Y,m,c)
plotting(m,c)
m,c = GradientDescent(X,Y,m,c)
plotting(m,c)
m,c = GradientDescent(X,Y,m,c)
plotting(m,c)

#*******************************USING LR EQUATION********************#

#mean of X,Y
x_mean = np.mean(X)
y_mean = np.mean(Y)

#Slope
m = np.sum((X-x_mean)*(Y-y_mean))/np.sum((X-x_mean)**2)

#Intercpt c
c = y_mean - (m*x_mean)

#regression line
print("y_cap = ",m,"X + ",c);

y_cap = []

#Predicted Y_cap values
for x in X:
  y_cap.append(m*x+c)

#list to array
y_cap = np.array(y_cap)

#plotting result
plt.title('Perfect LR Equation ')
plt.scatter(X, Y, color='b', label='Actual Data')
plt.plot(X,y_cap,"g-")
plt.legend()
plt.show()

#MSE Of Trainig data
temp = np.sum((Y-y_cap)**2)
MSE_training = temp/len(Y)
print('MSE of Training Set',MSE_training)

#y_cap Of testing data
y_cap_test = []
for x in X_test:
  y_cap_test.append(m*x+c)

y_cap_test = np.array(y_cap_test)

#plot testing data with predicted output
plt.title('Testing Data plot ')
plt.scatter(X_test,Y_test,color='b',label='Actual Data')
plt.plot(X_test,y_cap_test,"y-")
plt.legend()
plt.show()

#MSE of testing data
MSE_testing = np.sum((Y_test-y_cap_test)**2)/len(Y_test)
print('MSE of Testing Data' , MSE_testing)
