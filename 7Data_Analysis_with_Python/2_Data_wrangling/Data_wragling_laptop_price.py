import pandas as pd
import numpy as np
import matplotlib.pylab as plt

filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"
df = pd.read_csv(filepath , header=None)


headers = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core",
"Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]

df.columns = headers 
print(df.head(3))
df.replace('?' , np.nan, inplace = True)

#Note that we can update the `Screen_Size_cm` column such that
#all values are rounded to nearest 2 decimal places by using `numpy.round()`
#df[['Screen_Size_cm']] = np.round(df[['Screen_Size_cm']],2)

df.replace('?', np.nan, inplace = True)
missing_data = df.isnull()
print(missing_data.head())
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print('')

avg_weight_kg = df["Weight_kg"].astype("float").mean(axis=0)
df["Weight_kg"].replace(np.nan, avg_weight_kg)

# replacing missing data with mode
common_screen_size = df['Screen_Size_inch'].value_counts().idxmax()
df["Screen_Size_inch"].replace(np.nan, common_screen_size)

#Both "Weight_kg" and "Screen_Size_cm" are seen to have the data type "Object",
#while both of them should be having a data type of "float". Write a code to fix
#the data type of these two columns.

df[["Weight_kg","Screen_Size_inch"]] = df[["Weight_kg","Screen_Size_inch"]].astype("float")

#Data standardization: convert weight from kg to pounds
df['Weight_kg'] = df['Weight_kg'] * 2.205
df.rename(columns = {'Weight_kg' : 'Weight_pounds'}, inplace =True)


#Data Normalization
df['CPU_frequency'] = df['CPU_frequency']/df['CPU_frequency'].max()


#Binning
#Binning is a process of creating a categorical attribute
#which splits the values of a continuous data into a 
#specified number of groups. In this case, write a code to
#create 3 bins for the attribute "Price". 
#These bins would be named "Low", "Medium" and "High". The new attribute will be named "Price-binned".


bins = np.linspace(min(df['Price']), max(df['Price']), 4 )
group_names = ['Low', 'Medium', 'High']
df['Price-binned'] = pd.cut(df['Price'], bins, labels=group_names, include_lowest =True)

#ploting
plt.bar(group_names, df["Price-binned"].value_counts())
plt.xlabel("Price")
plt.ylabel("count")
plt.title("Price bins")

#Indicator variables
dummy_variable_1 = pd.get_dummies(df["Screen"])
dummy_variable_1.rename(columns={'IPS Panel':'Screen-IPS_panel', 'Full HD':'Screen-Full_HD'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "Screen" from "df"
df.drop("Screen", axis = 1, inplace=True)

print(df.head())