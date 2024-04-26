import numpy as np  
import pandas as pd 
df_can = pd.read_excel(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx',
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)
print(df_can.head(3))
df_can.info(verbose=False)
df_can.columns
df_can.index

print(type(df_can.columns))
print(type(df_can.index))

#To get the index and columns as lists, we can use the `tolist()` method.
print(type(df_can.columns.tolist()))
print(type(df_can.index.tolist()))
#size of dataframe (rows, columns)
df_can.shape

#Let's clean the data set to remove a few unnecessary columns. We can use *pandas* `drop()` method as follows:
df_can.drop(['AREA','REG','DEV','Type','Coverage',], axis=1, inplace=True)
print(df_can.head(2))

#Let's rename the columns so that they make sense. We can use `rename()` method by passing in a dictionary of old and new names as follows:
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
print(df_can.columns)
print(df_can)
#We will also add a 'Total' column that sums up the total immigrants by country over the entire period 1980 - 2013, as follows:
#df_can['Total'] = df_can.sum(axis=1)
#print(df_can)
#df_can.isnull().sum()
df_can.describe()

print(df_can.Country)
print(df_can[['Country', 1980, 1981, 1982, 1983, 1984, 1985]]) # returns a dataframe
# notice that 'Country' is string, and the years are integers. 
# for the sake of consistency, we will convert all column names to string later on.

# Select Row
#There are main 2 ways to select rows:
#df.loc[label]    # filters by the labels of the index/column
#print(df_can.iloc['Country'])   # filters by the positions of the index/column

df_can.set_index('Country', inplace=True)
# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()
print(df_can.head(3))

# optional: to remove the name of the index
#df_can.index.name = None

# 1. the full row data (all columns)
print(df_can.loc['Japan'])
df_can[df_can.index == 'Japan']
print(df_can.loc['Japan', 2013])

# 3. for years 1980 to 1985
print(df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1984]])
# Alternative Method
#df_can.iloc[87, [3, 4, 5, 6, 7, 8]]

#print(df_can.loc['Haiti'])
print(df_can[df_can.index == 'Haiti'])
print(df_can.loc['Haiti',2000])
print(df_can.loc['Haiti',[1990,1991,1992,1993,1994,1995]])

#Filtering based on a criteria <a id="4"></a>
#To filter the dataframe based on a condition, we simply pass the condition as a boolean vector. 
#For example, Let's filter the dataframe to show the data on Asian countries (AreaName = Asia).

# 1. create the condition boolean series
condition = df_can['Continent'] == 'Asia'
print(condition)
# 2. pass this condition into the dataFrame
df_can[condition]

# we can pass multiple criteria in the same line.
# let's filter for AreaNAme = Asia and RegName = Southern Asia

df_can[(df_can['Continent']=='Asia') & (df_can['Region']=='Southern Asia')]

# note: When using 'and' and 'or' operators, pandas requires we use '&' and '|' instead of 'and' and 'or'
# don't forget to enclose the two conditions in parentheses
print(df_can[(df_can['Continent']== 'Africa') & (df_can['Region'] == 'Southern Africa') ])

df_can.sort_values(by='Country', ascending=True, axis=0, inplace=True)
top_5 = df_can.head(5)
print(top_5)
#df.sort_values(col_name, axis=0, ascending=True, inplace=False, ignore_index=False)```<br><br>
#col_nam - the column(s) to sort by. <br>
#axis - axis along which to sort. 0 for sorting by rows (default) and 1 for sorting by columns.<br>
#ascending - to sort in ascending order (True, default) or descending order (False).<br>
#inplace - to perform the sorting operation in-place (True) or return a sorted copy (False, default).<br>
#ignore_index - to reset the index after sorting (True) or keep the original index values (False, default).<br>
df_can.sort_values(by = 1995, ascending = True, axis =0, inplace =True)
top3_2010 = df_can[1995].head(3)
print(top3_2010)

#let's convert the column names into strings: '1980' to '2013'.
df_can.columns = list(map(str, df_can.columns))
years = list(map(str, range(1980,2014)))
df_can.sort_values(by = '1995', ascending = True, axis =0, inplace =True)
top3_2010 = df_can['1995'].head(3)
print(top3_2010)
haiti = df_can.loc['Haiti', years]
print(haiti)

#df_can['Total'] = df_can.sum(axis = 1)
df_can['Total'] = df_can.loc[ :,'1980':'2013'].sum(axis=1)
total = df_can['Total']
print(total)



years = list(map(str, range(1980,2014))) 
df_can.sort_values(['Total',], ascending = False, axis = 0 , inplace = True )
top_3 = df_can.Total.head(3)


