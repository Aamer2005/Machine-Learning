import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor , plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv('employee_salary_dataset.csv')

X = df[['Years_of_Experience','Education_Level','Working_Hours_per_Week']]
Y = df['Salary']

model = DecisionTreeRegressor(max_depth=3,random_state=42)
model.fit(X,Y)

plt.figure(figsize=(20,10))
plot_tree(model,feature_names=X.columns,filled=True,rounded=True,fontsize = 12)
plt.show()