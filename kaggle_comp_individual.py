import numpy as np
import pandas as pd
import sklearn as skl
from statsmodels import tools
from sklearn import linear_model, preprocessing
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import math


##### load data and group obscure countries and professions into "other" values #####

df_train = pd.read_csv('/Users/Christopher/Desktop/HPC/ML/kaggle comp/training_data.csv')
df_test = pd.read_csv('/Users/Christopher/Desktop/HPC/ML/kaggle comp/test_data.csv').drop(['Income'],axis=1)
df_train = df_train[(df_train['Income in EUR']<2000000) & (df_train['Income in EUR']>3000)]
df_train['Training'] = 1
df_test['Training'] = 0

df_amal = pd.concat([df_train, df_test],sort=False).reset_index(drop=True)

for variable in ['Country','Profession']:
    if variable == 'Country': bar = 10
    else: bar = 30
    temp = df_amal[[variable]].copy()
    temp['count'] = 1
    temp = temp.groupby([variable]).sum().sort_values(['count'],ascending=False)
    others = temp[temp['count']<bar].index.tolist()
    df_amal[variable] = df_amal[variable].apply(lambda x: 'other_{}'.format(variable) if x in others else x)

df_amal = df_amal.merge(prof_means, how='left', on=['Profession'])
df_amal = df_amal.drop(['Profession'],axis=1)
df_amal = df_amal.rename({'Profession_short':'Profession'},axis=1)


##### visual exploration of data #####

X_train['log_gdp'] = X_train['GDP'].apply(lambda x: np.log(x))
for variable in ['Year of Record','Age','log_city','Body Height [cm]']:
    print(variable)
    plt.scatter(x=X_train[variable],y=df_train['Income in EUR'])
    plt.show()
    
    
##### impute missing values and create non-linear variabels #####

for variable in ['Gender','University Degree','Profession','Age','Year of Record']:
    df_amal[variable] = df_amal[variable].fillna(0)
    if variable not in ['Age','Year of Record']: df_amal[variable] = df_amal[variable].replace({'unknown':0,'0':0})

    if variable in ['Age','Year of Record']: strategy = 'mean'
    else: strategy = 'most_frequent'
    impute = SimpleImputer(missing_values=0, strategy=strategy)
    impute.fit(df_amal[[variable]])
    df_amal[variable] = impute.transform(df_amal[[variable]])
    
df_amal['Year of Record'] = df_amal['Year of Record'] - 1980
df_amal['log_city'] = df_amal['Size of City'].apply(lambda x: math.log(x))
df_amal = df_amal.drop(['Size of City'],axis=1)
df_amal['city_sqr'] = df_amal['log_city']**2
df_amal['age_sqr'] = df_amal['Age']**2
df_amal['height_sqr'] = df_amal['Body Height [cm]']**2


##### create dummy variables for categorical variables #####

for variable in ['University Degree','Gender','Country','Profession']:
    enc = preprocessing.OneHotEncoder()
    enc.fit(df_amal[[variable]])
    dummies = pd.DataFrame(enc.transform(df_amal[[variable]]).toarray())
    dummies.columns = list(enc.categories_[0])
    df_amal = df_amal.join(dummies)
    
    
##### define training and test sets #####

X_train = df_amal[df_amal['Training']==1].drop(['Instance','Gender','Country','Profession','University Degree',
                                                'Hair Color','other','Austria','other_Profession','No',
                                                'Training','Income in EUR'],axis=1)
y_train = df_amal[df_amal['Training']==1][['Income in EUR']]
X_test = df_amal[df_amal['Training']==0].drop(['Instance','Gender','Country','Profession','University Degree',
                                                'Hair Color','other','Austria','other_Profession','No',
                                                'Training','Income in EUR'],axis=1)    
                                                
 
 ##### train model using linear regression #####

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

pred = regr.predict(X_test)
attempt = pd.DataFrame(pred, columns=['Short Professions'])


##### choose m most relevant variables #####

rfe = RFE(estimator=regr, n_features_to_select=800, step=15)
rfe.fit(X_train, y_train)

ranks = rfe.ranking_
ranks = pd.DataFrame(ranks,columns=['rank'])
ranks['variable'] = X_train.columns.tolist()
ranks.sort_values(by=['rank'])

rel_vars = (rfe.support_ * np.array(X_train.columns)).tolist()
rel_vars = [x for x in rel_vars if x!='']
X_train = X_train[rel_vars]
X_test = X_test[rel_vars]


##### categorise similar professions #####

df_train = pd.read_csv('/Users/Christopher/Desktop/HPC/ML/kaggle comp/training_data.csv')
df_train['Profession'] = df_train['Profession'].str.lower()

jobs = ['officer','clerk','administrator','assistant','agent','staff','manager','analyst','engineer','developer',
        'supervisor','inspector','aide','executive','specialist','coordinator','driver','installer','auditor',
       'designer','attorney','caretaker','programmer']

def categorise(x):
    words = x.split()
    for job in jobs:
        if job in words: x=job
    return x

prof_means['Profession_short'] = prof_means['Profession'].apply(categorise)
prof_means['Profession_short'] = prof_means['Profession_short'].apply(lambda x: x.replace(' ',''))
