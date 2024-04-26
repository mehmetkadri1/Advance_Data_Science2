import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import seaborn

filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(filepath, header = None) 
df.columns = headers 
#print(df.head(3))

df.replace ('?', np.nan , inplace = True)

#Evaluating for missing data. You can use two methods
# .isnull()  or .notnull()
missing_data = df.isnull ()
#count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")  


#Deal with missing data 
#Drop data
    #Drop the whole row<br>
    #Drop the whole column
    #Replace data
        #Replace it by mean
        #Replace it by frequency
        #Replace it based on other function


#***Calculate the mean value then replace 'NaN' with mean value on column***
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
#Replace "NaN" with mean value in "normalized-losses" column
df['normalized-losses'] = df['normalized-losses'].fillna(avg_norm_loss)
avg_bore=df['bore'].astype('float').mean(axis=0)
df["bore"].replace(np.nan, avg_bore)
#Calculate the mean value for the 'bore' column
avg_stroke = df['stroke'].astype('float').mean(axis=0)
df["stroke"].replace(np.nan, avg_stroke)
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_horsepower)
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_peakrpm)

###ex: mean=df['bedrooms'].mean()
#df['bedrooms'].replace(np.nan,mean, inplace=True)


df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
df["num-of-doors"].replace(np.nan, "four")


#Finally, drop all rows that do not have price data:
df.dropna(subset = ['price'], axis = 0, inplace = True)
#reset index because we droped two rows
df.reset_index(drop = True, inplace =True)


#Correct data format.Making sure that all data is in correct format
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df[['price']] = df[['price']].astype('float')
df[['horsepower']] = df[['horsepower']].astype('float')
#df[['fuel-system']] = df[['fuel-system']].astype('float')
#df[['num-of-cylinders']] = df[['num-of-cylinders']].astype('float')
#df[['engine-type']] = df[['engine-type']].astype('float')

#Good! Now, you have a data set with no missing values. 
#The last step of data cleaning is making sure that all data is correct format.
print(df.dtypes)




#***Data Standardization***
#for example transform mpg to L/100 km
#Use this formula for unit convertion L/100 km = 235 / mpg
df['city-L/100km'] = 235/df["city-mpg"]
df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns = {'highway-mpg':'highway-L/100km'} , inplace = True)

#***Data Normalization***
# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
print(df[["length","width","height"]].head())

#***Binning***
#Since you are building 3 bins you need 4 dividers
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df["horsepower-binned"].value_counts()
#print(df[['horsepower','horsepower-binned']].head(20))

plt.hist(df['horsepower'], bins = 3)
#plt.bar(group_names, df["horsepower-binned"].value_counts())
# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#plt.show()


#***Indicator variable*** Regression doesnt understand words only numbers
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
#print(dummy_variable_1.head())

# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)
# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

dummy_variable_2 = pd.get_dummies(df['aspiration'])
#print(dummy_variable_2.head())
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop('aspiration', axis=1, inplace = True)


#Save the new csv 
df.to_csv('clean_df.csv')


















### 3 EXPLORATORY DATA ANALYSIS###

print(df.dtypes)
data_type = df['peak-rpm'].dtypes
print("Data type of 'peak-rpm' column:", data_type)
#we can calculate the correlation between variables of type 'int64
#or float64 using method corr
#df.corr()
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()


#In order to start understanding the (linear) relationship 
#between an individual variable and the price, 
#we can use "regplot" which plots the scatterplot plus
# the fitted regression line for the data. 
#This will be useful later on for visualizing the fit of the 
#simple linear regression model as well.
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0)
#plt.show()
print(df[["engine-size", "price"]].corr()) #reliable corr is 0.87
sns.regplot(x="peak-rpm", y="price", data=df)#it is not reliable variable
plt.ylim(0)
#plt.show()
#print(df[['peak-rpm','price']].corr()) correlation between is -0.1016
#print(df[['stroke', 'price']].corr())
sns.regplot(x="stroke", y="price", data=df)#it is not reliable corr is low
plt.ylim(0)
#plt.show()

#***Categorical variables**
sns.boxplot(x="body-style", y="price", data=df)#it is not good predictor
plt.ylim(0)
#plt.show()
sns.boxplot(x="engine-location", y="price", data=df)#it is good predictor
plt.ylim(0)
#plt.show()
sns.boxplot(x="drive-wheels", y="price", data=df)#it is good predictor
plt.ylim(0)
#plt.show()
print(df.describe())


#***Value counts***
#it is a good way of undestanding how many units of each 
#characteristic/variable we have.
df.describe(include = ['object'])
print(df['drive-wheels'].value_counts())
df['drive-wheels'].value_counts().to_frame()

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

# engine-location as variable
#result is front = 198 rear = 3 so it means engine location is not good predictor
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))

#***Basics of Grouping***
# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
print(grouped_pivot)

df_gptest2 = df[['body-style', 'price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
print(grouped_test_bodystyle)




#use the grouped results
#plt.pcolor(grouped_pivot, cmap='RdBu')
#plt.colorbar()
#plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
#plt.show()







#***Correlation and Causation ***
#1 is perfect positive linear correlatin
#0 is no linear correlation, the two variables most likely 
#do not affect each other
#-1 perfect negative linear correlation 
#df.corr()

'''''
P-value
What is this P-value? The P-value is the probability value 
that the correlation between these two variables is statistically significant
Normally, we choose a significance level of 0.05,
which means that we are 95% confident that the correlation between
 the variables is significant.

p-value is $<$ 0.001:
we say there is strong evidence that the correlation is significant.
the p-value is $<$ 0.05:
there is moderate evidence that the correlation is significant.
the p-value is $<$ 0.1: 
there is weak evidence that the correlation is significant.
the p-value is $>$ 0.1: 
there is no evidence that the correlation is significant.
'''''

#Let's calculate the  Pearson Correlation Coefficient and
#P-value of 'wheel-base' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
#Since the p-value is $<$ 0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong 
#pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 
#Since the p-value is $<$ 0.001, the correlation between horsepower and price is statistically significant, and the linear relationship is quite strong (~0.809, close to 1) 
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
#Since the p-value is $<$ 0.001, the correlation between length and price is statistically significant, and the linear relationship is moderately strong (~0.691)
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 
#Since the p-value is < 0.001, the correlation between width and price is statistically significant, and the linear relationship is quite strong (~0.751)
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
#Since the p-value is $<$ 0.001, the correlation between curb-weight and price is statistically significant, and the linear relationship is quite strong (~0.834).
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
#Since the p-value is $<$ 0.001, the correlation between engine-size and price is statistically significant, and the linear relationship is very strong (~0.872).
#pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 
#Since the p-value is $<$ 0.001, the correlation between bore and price is statistically significant, but the linear relationship is only moderate (~0.521).
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
#Since the p-value is $<$ 0.001, the correlation between city-mpg and price is statistically significant, and the coefficient of about -0.687 shows that the relationship is negative and moderately strong.
#pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
#print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 
#Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant, and the coefficient of about -0.705 shows that the relationship is negative and moderately strong.

'''''
We now have a better idea of what our data looks like and which variables are important to take into account when predicting the car price. We have narrowed it down to the following variables
Continuous numerical variables:Length Width Curb-weight Engine-size Horsepower City-mpg Highway-mpg Wheel-base Bor
Categorical variables:Drive-wheels
As we now move into building machine learning models to automate our analysis, feeding the model with variables that meaningfully affect our target variable will improve our model's prediction performance.
'''




























# 4MODEL DEVELOPMENT #

#In this section, we will develop several models that will predict the price of the car using the variables or features. This is just an estimate but should give us an objective idea of how much the car should cost.
#Some questions we want to ask in this module, Do I know if the dealer is offering fair value for my trade-in?, Do I know if I put a fair value on my car?
'''''
Linear Regression and Multiple Linear Regression
One example of a Data  Model that we will be using is
Simple Linear Regression

Simple Linear Regression is a method to help us understand the relationship between two variables:

The predictor/independent variable 
The response/dependent variable (that we want to predict)

The result of Linear Regression is a linear function that predicts the response (dependent) variable as a function of the predictor (independent) variable.

Y: Response \ Variable
X: Predictor \ Variables
Yhat = a + b  X
'''''
from sklearn.linear_model import LinearRegression

#Create the linear regression object
lm = LinearRegression()
X = df[['highway-L/100km']]
Y = df['price']
lm.fit(X,Y)
#we can output a prediction
Yhat=lm.predict(X)
Yhat[0:5]  
#what is the value of the intercept a?
lm.intercept_
#what is the value of the intercept b?
lm.coef_
#linear model is  Price = 38423.31 - 821.73 x highway-mpg


#***Multiple Linear Regression***
#If we want to use more variables in our model to predict car price, we can use <b>Multiple Linear Regression</b>.
#Multiple Linear Regression is very similar to Simple Linear Regression, but this method is used to explain the relationship between one continuous response (dependent) variable and <b>two or more</b> predictor (independent) variables.
#Most of the real-world regression models involve multiple predictors. We will illustrate the structure by using four predictor variables, but these results can generalize to any integer:

#Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
Z = df[['wheel-base', 'engine-size', 'highway-L/100km']]
lm.fit(Z, df['price'])
lm.intercept_
lm.coef_

#Regression Plot: When it comes to simple linear regression, an excellent way to visualize the fit of our model is by using regression plots
#One thing to keep in mind when looking at a regression plot is to pay attention to how scattered the data points are around the regression line. This will give you a good indication of the variance of the data and whether a linear model would be the best fit or not. If the data is too far off from the line, this linear model might not be the best model for this data.
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-L/100km", y="price", data=df)
plt.ylim(0)
#plt.show()


print(df[["peak-rpm","highway-L/100km","price"]].corr())


'''''
Residual Plot

A good way to visualize the variance of the data is to use a residual plot.
What is a residual
The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e). When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.
So what is a residual plot
A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.
What do we pay attention to when looking at a residual plot?
We look at the spread of the residuals:
 If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate for the data.

Why is that? Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data.

'''''
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-L/100km'], y=df['price'])
#plt.show()


#We can see from this residual plot that the residuals are not randomly spread around the x-axis, leading us to believe that maybe a non-linear model is more appropriate for this data

#Multilinear regression residiul plot
#Y_hat = lm.predict(Z)
#plt.figure(figsize=(width, height))
#ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
#sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
#plt.title('Actual vs Fitted Values for Price')
#plt.xlabel('Price (in dollars)')
#plt.ylabel('Proportion of Cars')
#plt.show()
#plt.close()




#*3.3Polynomial Regression and Pipelines
#Polynomial regression is aparticular case of the general linear regression model or multiple linear regression models.
#We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.
'''''def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-L/100km']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'highway-L/100km')
np.polyfit(x, y, 3)

ex2:
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'Highway MPG')
'''''

#We can perform a polynomial transform on multiple features. First, we import the module:
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
Z_pr=pr.fit_transform(Z)
Z.shape
Z_pr.shape



#***PIPELINE***
#Data Pipelines simplify the steps of processing the data.
#We use the module Pipeline to create a pipeline.
# We also use <b>StandardScaler as a step in our pipeline.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
#First, we convert the data type Z to type float to avoid conversion warnings that may appear as a result of StandardScaler taking float inputs.
#Then, we can normalize the data,  perform a transform and fit the model simultaneously.
Z = Z.astype(float)
pipe.fit(Z,Y)
#Similarly,  we can normalize the data, perform a transform and produce a prediction  simultaneously.
ypipe=pipe.predict(Z)
#print(r2_score(Y,ypipe))
ypipe[0:4]

'''''
Create a pipeline that standardizes the data, 
then produce a prediction using a linear regression model using the features Z and target
Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]

'''''








#***3.4Measure for In-Sample Evaluation ***

#WHen evaluating our models, not only do we want to visualize the results,
#but we also want a quantitative measure to determine how accurate the model is.
#Two very important measures:
#R^2 / R-squared  is coefficient of determination, measure to indicate how close the data is to the fitted regression line.
#Mean Squared Error (MSE) measures the avarage of the squares of errors.
#That is, the difference between actual value(y) and the estimated value (ŷ).
#The model with higher R squared value is a better fit for data
#The model with the smalles MSE value is a better fit for the data. 

### Model 1: Simple Linear Regression

#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))
#We can predict the output i.e 'yhat' using the predict method, where X is the input variable.
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)


### Model 2: Multiple Linear Regression

# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))
#We can say that 80 percent of the variation of price is explained by this 
Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

### Model 3: Polynomial Fit

#from sklearn.metrics import r2_score
#r_squared = r2_score(Y, p(x))
#print('The R-square value is: ', r_squared)
#mean_squared_error(df['price'], p(x))




# 3.5 Prediction and Decision Making

#Create new input
#new_input=np.arange(1, 100, 1).reshape(-1, 1)
#Fit the model
#lm.fit(X, Y)
#Produce a prediction
#yhat=lm.predict(new_input)
#yhat[0:5]
#plt.plot(new_input, yhat)
#plt.show()

#CONCLUSION DECISION MAKING: DETERMINING A GOOD MODEL FIT

#We already know R-squared and MSE values for the fits, how do we determine a good model fit?
#Lets take a look at the values for the differenct models.
#Model1 = R-s is 0.4 and MSE is 3.16 x 10(7)
#Model2 = R-s is 0.8 and MSE is 1.2 x 10(7)
#Model3 = R-s is 0.67 and MSE is 2.05 x 10(7)
#In order to compore the result look at a combination of both the R-s and MSE to make a best conclusion about the fit of the model.

########  Comparing these three models, we conclude that the MLR, it means model 2 is the best model to be able to predict price from our dataset.
#This result makes sense since we have 27 variables in total and we know that more than one of those variables are potential predictors of the final car price.




















### 5 MODEL EVALUATION ###

df=df._get_numeric_data()
print(df.head())

#Now, we define the functions that will be used in the lab for making plots at different stages.
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()





#An important step in testing your model is to split your data into training and testing data. We will place the target data price in a separate dataframe.
y_data = df['price']
#Drop price data in dataframe x_data
x_data=df.drop('price',axis=1)
#After executing this code, x_data will contain all the features except the 'price' column, and y_data will contain the target variable 'price'. 
#You can then use x_data and y_data for modeling tasks such as training a machine learning model.

#Now, we randomly split our data into training and testing data using the function train_test_split 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
print("number of test samples :", x_test1.shape[0])
print("number of training samples:",x_train1.shape[0])



lre=LinearRegression()
#we fit model using  features highway-L/100km
lre.fit(x_train[['highway-L/100km']], y_train)
#Let's calculate the R^2 on the test data:
print(lre.score(x_test[['highway-L/100km']], y_test))
#We can see the R^2 is much smaller using the test data compared to the training data.
print(lre.score(x_train[['highway-L/100km']], y_train))
lre.fit(x_train1[['highway-L/100km']],y_train1)
lre.score(x_test1[['highway-L/100km']],y_test1)


#Sometimes you do not have sufficient testing data; as a result, you may want to perform cross-validation. Let's go over several methods that you can use for cross-validation. 
#Cross-Validation Score 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#cv determine number of fold. In this case is 4
Rcross = cross_val_score(lre, x_data[['highway-L/100km']], y_data, cv=4)
#The default scoring is R^2. Each element in the array has the average R^2 value for the fold:
Rcross
#We can calculate the average and standard deviation of our estimate:
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
#We can use negative squared error as a score by setting the parameter  'scoring' metric to 'neg_mean_squared_error'. 
-1 * cross_val_score(lre,x_data[['highway-L/100km']], y_data,cv=4,scoring='neg_mean_squared_error')
#Calculate the average R^2 using two folds, then find the average R^2 for the second fold utilizing the "highway-L/100km" feature: 
Rc=cross_val_score(lre,x_data[['highway-L/100km']], y_data,cv=2)
Rc.mean()

#We input the object, the feature <b>"horsepower"</b>, and the target data <b>y_data</b>. The parameter 'cv' determines the number of folds. In this case, it is 4. We can produce an output:
yhat = cross_val_predict(lre,x_data[['highway-L/100km']], y_data,cv=4)
yhat[0:5]








### PART 2 Overfitting Underfitting and Model Selection