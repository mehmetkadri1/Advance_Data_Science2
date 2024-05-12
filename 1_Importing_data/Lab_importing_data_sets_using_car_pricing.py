import pandas as pd 
import numpy as np

file_path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv'
df = pd.read_csv(file_path, header=None)
#print('The first 5 rows of the dataframe')
#print(df.head(5))
#print(df.tail(10))

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df.columns = headers
df.columns
#print(df.head(5))

df1 = df.replace('?' , np.NaN)
#you can drop missing values along the column 'price' as follows:
df = df1.dropna(subset = ['price'], axis=0)
#Here, `axis=0` means that the contents along the entire row will be dropped
#wherever the entity 'price' is found to be NaN

#print(df.head(3))
#print(df.columns)

df.to_csv('automobile.csv', index = False)

#df.describe()
df[['length', 'compression-ratio']].describe()
