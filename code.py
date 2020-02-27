#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required libraries to perform calculations, build models and make plots.
import numpy as np
import pandas as pd
# import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


#Reading data from a file and creating a dataframe in pandas
data = pd.read_csv('Pima_Indian_diabetes.csv')


# # Exploratory Data Analysis

# In[3]:


'''
This is utility function it is used to plot density histogram of the given data.
'''
def plotDensityHistogram(data):
    sns.distplot(data, hist=True, kde=True, bins=int(180/5), color = 'darkblue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4})


# In[4]:


data.describe()


# In[5]:


#Calculating the no of NaN values in the data set and plotting the count
preg_nancount = len(data[data.Pregnancies.isna()].index)
st_nancount = len(data[data.SkinThickness.isna()].index)
bp_nancount = len(data[data.BloodPressure.isna()].index)
glu_nancount = len(data[data.Glucose.isna()].index)
in_nancount = len(data[data.Insulin.isna()].index)
bmi_nancount = len(data[data.BMI.isna()].index)
dpf_nancount = len(data[data.DiabetesPedigreeFunction.isna()].index)
age_nancount = len(data[data.Age.isna()].index)
nan_count = {}
nan_count['Pregnancy'] = preg_nancount
nan_count['Gluc'] = glu_nancount
nan_count['BP'] = bp_nancount
nan_count['ST'] = st_nancount
nan_count['Insu'] = in_nancount
nan_count['BMI'] = bmi_nancount
nan_count['DPF'] = dpf_nancount
nan_count['age'] = age_nancount
nan_count

nan_c = pd.DataFrame.from_dict(nan_count, orient = 'index')
nan_c_values = nan_c[0]
nan_c
plt.bar(nan_c.index, height = nan_c[0])
plt.title('Missing Values Histogram')
plt.xlabel('Parameter')
plt.ylabel('No of Missing Values')
plt.show()


# 

# In[6]:


data['Pregnancies'].plot.kde(title = "Pregnancy density distribution")


# In[7]:


data['Glucose'].plot.kde(title = "Glucose density distribution")


# In[8]:


data['BloodPressure'].plot.kde(title = "BloodPressure density distribution")


# In[9]:


data['SkinThickness'].plot.kde(title = "SkinThickness density distribution")


# In[10]:


data['Insulin'].plot.kde(title = "Insulin density distribution")


# In[11]:


data['BMI'].plot.kde(title = "BMI density distribution")


# In[12]:


data['Age'].plot.kde(title = "Age density distribution")


# In[13]:


data['DiabetesPedigreeFunction'].plot.kde(title = "DiabetesPedigreeFunction density distribution")


# In[14]:


plotDensityHistogram(data["Pregnancies"].dropna(how = 'all'))


# In[15]:


plotDensityHistogram(data["Glucose"].dropna(how = 'all'))


# In[16]:


plotDensityHistogram(data["BloodPressure"].dropna(how = 'all'))


# In[17]:


plotDensityHistogram(data["SkinThickness"].dropna(how = 'all'))


# In[18]:


plotDensityHistogram(data["Insulin"].dropna(how = 'all'))


# In[19]:


plotDensityHistogram(data["BMI"].dropna(how = 'all'))


# In[20]:


plotDensityHistogram(data["Age"].dropna(how = 'all'))


# In[21]:


plotDensityHistogram(data["DiabetesPedigreeFunction"].dropna(how = 'all'))


# # Data PrePreocessing

# In[22]:


#making all the ilegal values of every features NaN in the dataframe
data.loc[data['SkinThickness'] <= 0, 'SkinThickness'] = np.NaN
data.loc[data['Pregnancies'] < 0, 'Pregnancies'] = np.NaN
data.loc[data['Glucose'] <= 0, 'Glucose'] = np.NaN
data.loc[data['BloodPressure'] <= 0, 'BloodPressure'] = np.NaN
data.loc[data['BMI'] <= 0, 'BMI'] = np.NaN
data.loc[data['Insulin'] <= 0, 'Insulin'] = np.NaN


# In[23]:


#making all the values of Pregnancies column NaN if it is not an integer
pregs = np.array(data['Pregnancies'])
for i in range(len(pregs)):
    if not pregs[i].is_integer():
        pregs[i] = np.NaN
data['Pregnancies'] =  pregs


# In[24]:


#Calculating the no of NaN/incorrect values in the data set and plotting the count
preg_nancount = len(data[data.Pregnancies.isna()].index)
st_nancount = len(data[data.SkinThickness.isna()].index)
bp_nancount = len(data[data.BloodPressure.isna()].index)
glu_nancount = len(data[data.Glucose.isna()].index)
in_nancount = len(data[data.Insulin.isna()].index)
bmi_nancount = len(data[data.BMI.isna()].index)
dpf_nancount = len(data[data.DiabetesPedigreeFunction.isna()].index)
age_nancount = len(data[data.Age.isna()].index)
nan_count = {}
nan_count['Preg'] = preg_nancount
nan_count['Glu'] = glu_nancount
nan_count['BP'] = bp_nancount
nan_count['ST'] = st_nancount
nan_count['Insu'] = in_nancount
nan_count['BMI'] = bmi_nancount
nan_count['DPF'] = dpf_nancount
nan_count['age'] = age_nancount
nan_count

nan_c = pd.DataFrame.from_dict(nan_count, orient = 'index')
nan_c_values = nan_c[0]
nan_c
plt.bar(nan_c.index, height = nan_c[0])
plt.title('Incorrrect Data')
plt.ylabel('Count')
plt.show()


# In[25]:


#Removing rows with NaN in BloodPressure
data = data.drop(data[data.BloodPressure.isna()].index, axis = 0)


# In[26]:


#Removing raws with NaN in Glucose
data = data.drop(data[data.Glucose.isna()].index, axis = 0)


# In[27]:


data.describe()


# In[28]:


#Correlation Matrix
data.corr()


# In[29]:


heat_map = sns.heatmap(data.corr(), annot = True)


# In[30]:


#Replacing NaN with mode or median in Pregnancies, BMI, Age
data['Pregnancies'] = data['Pregnancies'].fillna(data['Pregnancies'].mode()[0])
data['BMI'] = data['BMI'].fillna(data['BMI'].mode()[0])                            
data['Age'] = data['Age'].fillna(data['Age'].mode()[0])


# In[31]:


plt.scatter(data['Glucose'], data['Insulin'] )
plt.title('Scatter plot Glucose vs Insulin')
plt.xlabel('Glucose')
plt.ylabel('Insulin')
plt.show()


# # Linear Regression

# In[32]:


'''
linearRegressionPrediction takes data, inp_fearures, out_features.
in_features are used to impute the missing values of out_features using multivarailte LinearRegression.
'''
def linearRegressionPrediction(data, inp_features, out_feature):
    #creating a copy of the actual data
    dat = data.copy()
    #droping all the data points whose out_feature attribute is NaN, this data will be used to fit linear regression model
    
    noNaNdata = dat.drop(dat[out_feature].index[dat[out_feature].apply(np.isnan)], axis = 0)
    
    #all the data points whose values of out_features are missing are stored in the variable tobepredicted 
    tobepredicted = dat[inp_features].loc[dat[out_feature].index[dat[out_feature].apply(np.isnan)]]
    
    
    df = pd.DataFrame(noNaNdata, columns = inp_features)
    target = pd.DataFrame(noNaNdata, columns = [out_feature])
    
    lm = linear_model.LinearRegression()
    
    model = lm.fit(df, target)
    
    #predicting the missing values of out_features in 'tobepredicted' .
    prediction_values = lm.predict(np.array(tobepredicted[inp_features]))
    
    #this function returns the predicted values for missing data and the corresponding indices of the midding values 
    return (np.matrix.transpose(np.array(prediction_values))[0], np.array(dat[out_feature].index[dat[out_feature].apply(np.isnan)]))


# # Missing Data Handling

# In[33]:


predicted_values, nanIdx = linearRegressionPrediction(data, ['BMI'], 'SkinThickness')
data['SkinThickness'].loc[nanIdx] = predicted_values
data.describe()


# In[34]:


predicted_values, nanIdx = linearRegressionPrediction(data, ['Glucose'], 'Insulin')
data['Insulin'].loc[nanIdx] = predicted_values
data.describe()


# # Feature Extraction

# # PCA

# In[35]:


'''
principalComponentAnalysis takes data and new dimensionality and returns a dataframe in a new featurespace with the dimensionality as specified.
'''
def principalComponentAnalysis(data, newDimensionality):
    #features are the existing features
    features = data.columns
    x = data.loc[:, features].values
    
    #standardising the data
    x = StandardScaler().fit_transform(x)
    
    #creating the  instance of PCA and fitting it with the standardised data
    pca = PCA(n_components = newDimensionality)
    principalComponents = pca.fit_transform(x)
    
    #creating a new dataframe based on transformed data
    data_pca = pd.DataFrame(data = principalComponents)
    return data_pca


# In[36]:


data_wo_oc = data.copy()
data_wo_oc = data_wo_oc.drop('Outcome', axis = 1)


# In[37]:


data_pca = principalComponentAnalysis(data_wo_oc, 3)
data_pca = data_pca.assign(Outcome = np.array(data['Outcome']))
data_pca


# # Model Building

# # Logistic Regression

# In[38]:


'''
datasplit is an utility function which ensures that after the split the training set has same ratio of instances
which are false and true as that in entire dataset.It is used to ensure that the split which happens is not biased
towards a particular outcome.
'''
def dataSplit(data, percent_test, outFeature, seed):
    np.random.seed(seed)
    lol_dat = data.copy().drop([outFeature], axis = 1)
    dat = pd.DataFrame(data[outFeature], columns  = [outFeature])
    dat1 = pd.DataFrame(dat[dat.Outcome == 1])
    dat0 = pd.DataFrame(dat[dat.Outcome == 0])
    len1 = dat1.shape[0]
    len2 = dat0.shape[0]
    percentToTrain = (1 - percent_test)
    dat1_shuffle = np.array(dat1.index)
    dat0_shuffle = np.array(dat0.index)
    np.random.shuffle(dat1_shuffle)
    np.random.shuffle(dat0_shuffle)
    a0 = dat1_shuffle[:int(percentToTrain*len1)]
    b0 = dat0_shuffle[:int(percentToTrain*len2)]
    a1 = dat1_shuffle[int(percent_test*len1):]
    b1 = dat0_shuffle[int(percent_test*len2):]
    train_idx = np.concatenate([a0, b0])
    test_idx = np.concatenate([a1, b1])
    X_train = lol_dat.loc[train_idx]
    X_test = lol_dat.loc[test_idx]
    y_train = data['Outcome'].loc[train_idx]
    y_test = data['Outcome'].loc[test_idx]
    return X_train, X_test, y_train, y_test


# In[39]:


'''
logisticRegression
we are using logisticRegression to accuratley classify the patient as diabetic or not.
This funtion takes data and tagetFeature(Outcome) and testPercent

data - the input data after datapreprocessing and feature extraction

targetFeature - the targetFeature which we are trying to predict

splitratio - the ratio of splitting data into testing and training sets

this function returns final report which contains training,testing sets and classification_report.
'''

def logisticRegression(data, tarFeature, splitratio):
    dat = data.copy()
    X_train, X_test, y_train, y_test = dataSplit(dat, splitratio, tarFeature, 3)
    print(y_train, y_test)
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    report = {}
    report['X_TRAIN'] = np.array(X_train)
    report['X_TEST'] = np.array(X_test)
    report['Y_TRAIN'] = np.array(y_train)
    report['Y_TEST'] = np.array(y_test)
    report['TRAIN_DATA_INDEX'] = np.array(X_train.index)
    report['TEST_DATA_INDEX'] = np.array(X_test.index)
    report['FINAL_REPORT'] = classification_report(y_test,predictions)
    return report


# In[40]:


final = logisticRegression(data_pca, 'Outcome', 0.25)
# final


# In[41]:


print(final['FINAL_REPORT'])

