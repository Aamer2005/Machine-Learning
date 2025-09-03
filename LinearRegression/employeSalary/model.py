import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('employee_salary_dataset.csv')

X = df[['Years_of_Experience','Education_Level','Working_Hours_per_Week']].values
Y = df['Salary'].values

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
Y_mean = Y.mean()
Y_std  = Y.std()

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

# ---------------- Training ----------------
iteration = 1000
learning_rate = 0.01
b0,b1,b2,b3 = 0,0,0,0
n = len(Y)
MSE = []

for _ in range(iteration):
    # Prediction
    y_pred = b0 + b1*X_norm[:,0] + b2*X_norm[:,1] + b3*X_norm[:,2]

    # Error 
    error = y_pred - Y_norm

    # Mean Squared Error
    avg_mean_square_error = np.mean(error**2)
    MSE.append(avg_mean_square_error)

    # Gradients
    db0 = (2/n)*np.sum(error)
    db1 = (2/n)*np.sum(error*X_norm[:,0])
    db2 = (2/n)*np.sum(error*X_norm[:,1])
    db3 = (2/n)*np.sum(error*X_norm[:,2])

    # Update
    b0 = b0 - learning_rate*db0
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2
    b3 = b3 - learning_rate*db3

print("Y = ",b0," + ",b1,"X1"," + ",b2,"X2"," + ",b3,"X3")

# ---------------- Prediction Function ----------------
def predict(years,edu,hours):
    years_n = (years - X_mean[0]) / X_std[0]
    edu_n   = (edu   - X_mean[1]) / X_std[1]
    hours_n = (hours - X_mean[2]) / X_std[2]

    y_norm =  b0 + b1*years_n + b2*edu_n + b3*hours_n
    y_original = y_norm*Y_std + Y_mean
    return y_original

# ---------------- MSE Plot ----------------
def MSE_plot():
    plt.plot(MSE,color='blue',label='MSE')
    plt.legend()
    plt.show()

MSE_plot()

print(predict(1,1,35))
print(predict(12,3,55))


# #we can convert b0,b1... so that the input is directly taken
# B0 = Y_mean + Y_std*b0 - Y_std*b1*X_mean[0]/X_std[0] - Y_std*b2*X_mean[1]/X_std[1] - Y_std*b3*X_mean[2]/X_std[2]
# B1 = Y_std*b1 / X_std[0]
# B2 = Y_std*b2 / X_std[1]
# B3 = Y_std*b3 / X_std[2]

# def predict_direct(years, edu, hours):
#     return B0 + B1*years + B2*edu + B3*hours
