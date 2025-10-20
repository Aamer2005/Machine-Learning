import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('employeeData.csv')

X = df[['Hours']]

Y = df['Score']

model = LinearRegression()
model.fit(X,Y)

predict = model.predict(X)

plt.scatter(X,Y,color='blue')
plt.plot(X,predict,"r-")
plt.legend()
plt.show()