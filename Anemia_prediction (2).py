#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing


# In[2]:


data = pd.read_csv("smallanemia.csv")


# In[3]:


data.head()


# In[4]:


# 0 - male and 1 - female 


# In[5]:


data.info()
#checking datatypes


# In[6]:


data.isnull().sum()


# In[7]:


#checking if the data is balanced or not (biasness)
f,ax=plt.subplots(1,2,figsize=(10,8))
data['Result'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('had anemia')
ax[0].set_ylabel('')
sns.countplot('Result',data=data,ax=ax[1])
ax[1].set_title('had anemia')
plt.show()


# f,ax=plt.subplots(1,2,figsize=(10,8)): This line creates a figure with two subplots arranged horizontally. f is the figure object, ax is a NumPy array with two axes (one for each subplot), and figsize specifies the size of the figure.
# 
# data['Result'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True): This line creates a pie chart of the "Result" column in the "data" DataFrame. value_counts() counts the number of occurrences of each unique value in the column, and plot.pie() plots the counts as a pie chart. explode=[0,0.1] specifies that the second slice of the pie should be "exploded" (i.e., pulled out) by 0.1 of the radius. autopct='%1.1f%%' formats the slice labels as percentages with one decimal place. ax=ax[0] specifies that the pie chart should be plotted in the first subplot. shadow=True adds a shadow effect to the chart.
# 
# ax[0].set_title('had anemia'): This line sets the title of the first subplot to "had anemia".
# 
# ax[0].set_ylabel(''): This line sets the y-label of the first subplot to an empty string.
# 
# sns.countplot('Result',data=data,ax=ax[1]): This line creates a count plot of the "Result" column in the "data" DataFrame. sns.countplot() is a function from the Seaborn library that creates a bar plot of the counts of each unique value in a column. ax=ax[1] specifies that the count plot should be plotted in the second subplot.
# 
# ax[1].set_title('had anemia'): This line sets the title of the second subplot to "had anemia".
# 
# plt.show(): This line displays the figure containing the two subplots with the pie chart and the count plot.
# 

# In[8]:


#groupwise distribution of results 
data.groupby(['Gender','Result'])['Result'].count()


# In[9]:


#genderwise anemia 
f,ax=plt.subplots(1,2,figsize=(10,7))
data[['Gender','Result']].groupby(['Gender']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Having Anemia vs Gender')
sns.countplot('Gender',hue='Result',data=data,ax=ax[1])
ax[1].set_title('Gender:Having Anemia vs Not Having Anemia')
plt.show()


# f,ax=plt.subplots(1,2,figsize=(10,7)): This line creates a figure with two subplots arranged horizontally. f is the figure object, ax is a NumPy array with two axes (one for each subplot), and figsize specifies the size of the figure.
# 
# data[['Gender','Result']].groupby(['Gender']).mean().plot.bar(ax=ax[0]): This line creates a bar chart of the mean value of the "Result" column for each unique value in the "Gender" column. data[['Gender','Result']] selects only the "Gender" and "Result" columns from the "data" DataFrame. groupby(['Gender']).mean() groups the data by the "Gender" column and computes the mean value of the "Result" column for each group. plot.bar() plots the means as a bar chart. ax=ax[0] specifies that the bar chart should be plotted in the first subplot.
# 
# ax[0].set_title('Having Anemia vs Gender'): This line sets the title of the first subplot to "Having Anemia vs Gender".
# 
# sns.countplot('Gender',hue='Result',data=data,ax=ax[1]): This line creates a count plot of the "Gender" column in the "data" DataFrame, where the counts are split by the "Result" column. sns.countplot() is a function from the Seaborn library that creates a bar plot of the counts of each unique value in a column. hue='Result' specifies that the counts should be split by the "Result" column. ax=ax[1] specifies that the count plot should be plotted in the second subplot.
# 
# ax[1].set_title('Gender:Having Anemia vs Not Having Anemia'): This line sets the title of the second subplot to "Gender: Having Anemia vs Not Having Anemia".
# 
# plt.show(): This line displays the figure containing the two subplots with the bar chart and the count plot.
# 
# Overall, this code is creating two visualizations that compare the incidence of anemia by gender. The first subplot shows the average incidence of anemia for each gender, while the second subplot shows the counts of anemia by gender, split by whether or not anemia is present.

# In[10]:


#working on hb column
print('The highest hemoglobin was of:',data['Hemoglobin'].max())
print('The lowest hemoglobin was of:',data['Hemoglobin'].min())
print('The average hemoglobin in the data:',data['Hemoglobin'].mean())


# Lower than normal hemoglobin levels indicate anemia. The normal hemoglobin range is generally defined as 13.2 to 16.6 grams (g) of hemoglobin per deciliter (dL) of blood for men and 11.6 to 15 g/dL for women.
# 
# 
# 
# 

# While hemoglobin (hb) level is an important factor in the diagnosis and classification of anemia, other factors such as MCV, MCHC, and red blood cell count can provide additional information about the underlying cause of anemia.
# 
# For example, the MCV measures the average size of red blood cells. If the MCV is low, it may indicate a type of anemia called microcytic anemia, which is often caused by iron deficiency or thalassemia. If the MCV is high, it may indicate a type of anemia called macrocytic anemia, which is often caused by vitamin B12 or folate deficiency.
# 
# Similarly, the MCHC measures the average concentration of hemoglobin in each red blood cell. If the MCHC is low, it may indicate a type of anemia called hypochromic anemia, which is often caused by iron deficiency. If the MCHC is high, it may indicate a type of anemia called hereditary spherocytosis.
# 
# Red blood cell count is another factor that can provide information about anemia. If the red blood cell count is low, it may indicate anemia, while a high red blood cell count may indicate a different condition such as polycythemia.
# 
# By considering these additional factors in addition to hemoglobin level, clinicians can more accurately diagnose and classify anemia, determine the underlying cause, and develop an appropriate treatment plan. Therefore, these other factors are important in the prediction of anemia.

# In[11]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['Result']==1]['Hemoglobin'].value_counts()
ax1.hist(data_len,color='blue')
ax1.set_title('Having anemia')
data_len=data[data['Result']==0]['Hemoglobin'].value_counts()
ax2.hist(data_len,color='violet')
ax2.set_title('NOT Having anemia')
fig.suptitle('Hemoglobin Levels')
plt.show()


# In[ ]:





# fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5)): This line creates a figure with two subplots (ax1 and ax2) side by side, and assigns the figure and the axes objects to fig and (ax1,ax2), respectively. figsize sets the size of the figure in inches.
# 
# data_len=data[data['Result']==1]['Hemoglobin'].value_counts(): This line creates a variable called data_len that stores the value counts of Hemoglobin for instances where the Result column is equal to 1 (i.e., the patient has anemia).
# 
# ax1.hist(data_len,color='red'): This line plots a histogram of the values in data_len on the first subplot (ax1) with the color set to red.
# 
# ax1.set_title('Having anemia'): This line sets the title of the first subplot to 'Having anemia'.
# 
# data_len=data[data['Result']==0]['Hemoglobin'].value_counts(): This line creates a new variable called data_len that stores the value counts of Hemoglobin for instances where the Result column is equal to 0 (i.e., the patient does not have anemia).
# 
# ax2.hist(data_len,color='green'): This line plots a histogram of the values in data_len on the second subplot (ax2) with the color set to green.
# 
# ax2.set_title('NOT Having anemia'): This line sets the title of the second subplot to 'NOT Having anemia'.
# 
# fig.suptitle('Hemoglobin Levels'): This line sets the title of the entire figure to 'Hemoglobin Levels'.
# 
# plt.show(): This line displays the figure.
# 
# Overall, this code creates a figure with two histograms side by side. The left histogram shows the distribution of Hemoglobin levels for patients with anemia, and the right histogram shows the distribution for patients without anemia. The color of the bars in each histogram is set to red and green, respectively, and the titles of each subplot reflect the corresponding anemia status.
# 
# 
# 
# 
# 
# 

#  MCH stands for “mean corpuscular hemoglobin.” An MCH value refers to the average quantity of hemoglobin present in a single red blood cell. Hemoglobin is the protein in your red blood cells that transports oxygen to the tissues of your body

# In[12]:


#MCh is mean corpuscular hb 
print('The highest MCH was of:',data['MCH'].max())
print('The lowest MCH was of:',data['MCH'].min())
print('The average MCH in the data:',data['MCH'].mean())


# In[13]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['Result']==1]['MCH'].value_counts()
ax1.hist(data_len,color='red')
ax1.set_title('Having anemia')
data_len=data[data['Result']==0]['MCH'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('NOT Having anemia')
fig.suptitle('MCH Levels')
plt.show()


# This code produces two histograms side-by-side using the subplots() function. The first line initializes two subplots named ax1 and ax2 and assigns them to the variables fig and ax1 and ax2, respectively. The subplots() function creates a figure with a single row and two columns to accommodate two subplots side-by-side. The figsize parameter sets the size of the figure.
# 
# The next three lines of code calculate the frequency of MCH values for patients with anemia and store the result in the variable data_len. This is done by first filtering the rows where Result equals 1 (i.e., patients with anemia), then selecting only the MCH column, and finally calculating the frequency of each unique value using the value_counts() method.
# 
# The fourth line of code creates a histogram of the data_len variable and plots it on the first subplot (ax1). The color parameter sets the color of the histogram bars to red, and the set_title() method sets the title of the subplot to "Having anemia".
# 
# The fifth and sixth lines of code repeat the process for patients without anemia. The only difference is that the Result column is filtered for a value of 0 instead of 1. The histogram of the data_len variable is plotted on the second subplot (ax2) with green bars, and the subplot title is set to "NOT Having anemia".
# 
# Finally, the suptitle() method sets the main title of the figure to "MCH Levels", and the show() function displays the figure. The resulting plot shows the distribution of MCH values for patients with and without anemia.

# The x-axis in both subplots represents the frequency of MCH values, and the y-axis represents the number of patients that have a certain frequency of MCH values.

# In[14]:


#"mean corpuscular hemoglobin concentration" (MCHC). MCHC checks the average amount of hemoglobin in a group of red blood cells
print('The highest MCHC was of:',data['MCHC'].max())
print('The lowest MCHC was of:',data['MCHC'].min())
print('The average MCHC in the data:',data['MCHC'].mean())


# In[15]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['Result']==1]['MCHC'].value_counts()
ax1.hist(data_len,color='red')
ax1.set_title('Having anemia')
data_len=data[data['Result']==0]['MCHC'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('NOT Having anemia')
fig.suptitle('MCHC Levels')
plt.show()


# In[16]:


# 5 aisi values hai jispe 8 logo ko anemia h  


# Mean corpuscular volume (MCV) is a laboratory value that measures the average size and volume of a red blood cell. It has utility in helping determine the etiology of anemia — calculation of the value is by multiplying the percent hematocrit by ten divided by the erythrocyte count.

# In[17]:


# See the min, max, mean values
print('The highest MCV was of:',data['MCV'].max())
print('The lowest MCV was of:',data['MCV'].min())
print('The average MCV in the data:',data['MCV'].mean())


# In[18]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['Result']==1]['MCV'].value_counts()
ax1.hist(data_len,color='red')
ax1.set_title('Having anemia')
data_len=data[data['Result']==0]['MCV'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('NOT Having anemia')
fig.suptitle('MCV Levels')
plt.show()


# After completing the EDA in above codes we are moving to data cleaning part 
# Feature Engineering and Data Cleaning
# We need to convert the continous values into categorical values by either Binning or Normalisation. Binning and normalisation both will be used in this section i.e group a range of ages into a single bin or assign them a single value.

# In[19]:


sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# In[20]:


from sklearn import preprocessing
import pandas as pd

d = preprocessing.normalize(data.iloc[:,1:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["Hemoglobin", "MCH", "MCHC", "MCV"])
scaled_df.head()


# The normalize() function scales the data in each column to have unit norm (i.e., a magnitude of 1) along the specified axis (in this case, axis=0 means that each column is normalized independently). This is a common technique for ensuring that the variables are on the same scale and that no single variable dominates the analysis.
# 
# The resulting normalized data is then stored in a new DataFrame called "scaled_df" with column names "Hemoglobin", "MCH", "MCHC", and "MCV".

# predictive modelling

# In[21]:


#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


# In[22]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Result'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=data[data.columns[:-1]]
Y=data['Result']
len(train_X), len(train_Y), len(test_X), len(test_Y) , len(X) , len(Y)
#train X consist of all features except the target and in y train we have target 


# 
# train_test_split(data, test_size=0.3, random_state=0, stratify=data['Result']): This function from the sklearn.model_selection module splits the data into training and testing sets. data is the dataset to be split, test_size=0.3 specifies that 30% of the data will be used for testing, random_state=0 ensures that the same random seed is used for the split every time the code is run, and stratify=data['Result'] ensures that the proportion of positive and negative samples in the data is preserved in the training and testing sets.
# train_X = train[train.columns[:-1]]: This line selects all columns except the last one (which is assumed to be the target variable) from the training data and assigns them to train_X.
# train_Y = train[train.columns[-1:]]: This line selects only the last column (which is assumed to be the target variable) from the training data and assigns it to train_Y.
# test_X = test[test.columns[:-1]]: This line selects all columns except the last one (which is assumed to be the target variable) from the testing data and assigns them to test_X.
# test_Y = test[test.columns[-1:]]: This line selects only the last column (which is assumed to be the target variable) from the testing data and assigns it to test_Y.
# X = data[data.columns[:-1]]: This line selects all columns except the last one (which is assumed to be the target variable) from the original data and assigns them to X.
# Y = data['Result']: This line selects only the last column (which is assumed to be the target variable) from the original data and assigns it to Y.
# len(train_X), len(train_Y), len(test_X), len(test_Y): This line prints the lengths of the training and testing sets for the user to verify.

# In[23]:


model=KNeighborsClassifier() 
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction1,test_Y))


# In[24]:


model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction2,test_Y))


# In[25]:


model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))


# Cross-validation is a statistical technique used in predictive analysis to evaluate the performance and generalization of a machine learning model. The goal of cross-validation is to estimate how well the model is likely to perform on new data that it has not seen during training.
# 
# In cross-validation, the dataset is divided into K subsets (or folds) of approximately equal size. The model is trained on K-1 folds and then tested on the remaining fold. This process is repeated K times, with each fold serving as the test set once. The results from each fold are then averaged to obtain an overall performance estimate.
# 
# One common type of cross-validation is K-fold cross-validation, where K is usually set to 5 or 10. In K-fold cross-validation, the dataset is randomly partitioned into K subsets, and the model is trained and tested K times, with each subset serving as the test set once. The average performance across the K folds is then used as the estimate of the model's generalization performance.
# 
# Another common type of cross-validation is leave-one-out cross-validation, where K is set to the number of samples in the dataset. In leave-one-out cross-validation, each sample is used as the test set once, and the model is trained on the remaining samples. The average performance across all samples is then used as the estimate of the model's generalization performance.
# 
# Cross-validation is a powerful tool for assessing the performance and generalization of machine learning models, as it provides an unbiased estimate of the model's performance on new data. It is often used in combination with other techniques such as hyperparameter tuning and feature selection to build robust and accurate machine learning models.

# In[26]:


#cross validation


# In[27]:


from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10) # k=10, split the data into 10 equal parts
cv_mean=[]
accuracy=[]
std=[]
classifiers=['Logistic Regression','KNN','Decision Tree']
models=[LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier()]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    cv_mean.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':cv_mean,'Std':std},index=classifiers)       
new_models_dataframe2


# In[28]:


#low std  show stability of data in case of different inputs dataset


# In[29]:


f,ax=plt.subplots(3,1,figsize=(12,10))
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=5),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0],annot=True,fmt='2.0f')
ax[0].set_title('Matrix for KNN')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1],annot=True,fmt='2.0f')
ax[1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2],annot=True,fmt='2.0f')
ax[2].set_title('Matrix for Decision Tree')
plt.show()


# In[30]:


from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=5)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(train_X,train_Y)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is :',cross.mean())


# Ensembling, also known as ensemble learning, is a technique in machine learning where multiple models are combined to improve the performance of a prediction task. The idea behind ensembling is to leverage the strengths of different models and combine their predictions to get a more accurate and robust prediction.
# 
# Ensembling can be done in various ways, but the two most common approaches are:
# 
# Bagging (Bootstrap Aggregating): In this approach, multiple models are trained on different subsets of the training data, with replacement. This means that some samples may appear in multiple subsets, while others may not appear at all. Each model then makes its prediction, and the final prediction is obtained by taking the average of the predictions of all models.
# 
# Boosting: In this approach, multiple models are trained sequentially, with each model trying to improve the errors of the previous model. In boosting, the models are trained on the same training set, but the weights of the samples are adjusted at each iteration to focus more on the samples that are misclassified by the previous model. The final prediction is obtained by weighting the predictions of all models based on their accuracy.
# 
# Ensembling is a powerful technique that can often lead to better performance than using a single model. It is widely used in various machine learning applications, including classification, regression, and clustering. Some popular ensemble methods include Random Forests, AdaBoost, Gradient Boosting Machines (GBMs), and Stacking.

# In[31]:


import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




