import pandas as pd;

df = pd.read_csv('data-employee.csv')

print(df.shape)

print(df['age'].max())

print(df.head(10))

print(df.columns)

print(df['age'])

df.to_csv("myFile.csv",index='False')

myage = df.groupby('age')
print(myage)

