import pandas as pd
import numpy as np

filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"
df = pd.read_csv(filepath , header=None)

print(df.head(5))

headers = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core",
"Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]

df.columns = headers 
print(df.head(3))
df.replace('?' , np.nan, inplace = True)

print(df.dtypes)
print(df.describe(include='all'))

print(df.info())

