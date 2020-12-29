#!/usr/bin/env python
# coding: utf-8

# In[11]:



import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
#from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
#from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
#from sklearn.cross_validation import StratifiedKFold 
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif,chi2
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import skew,kurtosis
import scipy.sparse as sp
import pickle as pkl
#!pip install catboost
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelBinarizer


# In[12]:


2


# In[15]:


def final_fun_1(data_1_test,session_df):
    data_1_test['date_account_created'] = pd.to_datetime(data_1_test['date_account_created'])
    data_1_test['timestamp_first_active'] = pd.to_datetime((data_1_test.timestamp_first_active // 1000000), format='%Y%m%d')
    data_1_test['weekday_account_created'] = data_1_test.date_account_created.dt.day_name()
    data_1_test['day_account_created'] = data_1_test.date_account_created.dt.day
    data_1_test['month_account_created'] = data_1_test.date_account_created.dt.month_name()
    data_1_test['year_account_created'] = data_1_test.date_account_created.dt.year
    data_1_test['weekday_first_active'] = data_1_test.timestamp_first_active.dt.day_name()
    data_1_test['day_first_active'] = data_1_test.timestamp_first_active.dt.day
    data_1_test['month_first_active'] = data_1_test.timestamp_first_active.dt.month_name()
    data_1_test['year_first_active'] = data_1_test.timestamp_first_active.dt.year
    data_1_test['time_gap']=(data_1_test['date_account_created'] - data_1_test['timestamp_first_active']).apply(lambda l: l.days)

    data_1_test['date_first_booking'] = pd.to_datetime(data_1_test['date_first_booking'])
    data_1_test['year_date_first_booking'] = data_1_test.date_first_booking.dt.year
    #age field which contain values in 19's series and date_first_booking is not null then will take the difference of the age and date_first_booking year.
    date_first_booking=data_1_test[(data_1_test.age>1000) & (data_1_test.age<2000) & ~data_1_test.date_first_booking.isnull() ][['year_date_first_booking']]
    age=data_1_test[(data_1_test.age>1000) & (data_1_test.age<2000) & ~data_1_test.date_first_booking.isnull() ][['age']]
    data_1_test.loc[((data_1_test.age>1000) & (data_1_test.age<2000) & (~data_1_test.date_first_booking.isnull())),'age' ]=date_first_booking['year_date_first_booking']-age['age']
    #************************************************************************************************************************************************************
    data_1_test['date_account_created'] = pd.to_datetime(data_1_test['date_account_created'])
    data_1_test['year_date_account_created'] = data_1_test.date_account_created.dt.year
    #age field which contain values in 19's series and date_account_created is not null then will take the difference of the age and date_account_created year.
    year_date_account_created=data_1_test[(data_1_test.age>1000) & (data_1_test.age<2000) & ~data_1_test.date_account_created.isnull() ][['year_date_account_created']]
    age=data_1_test[(data_1_test.age>1000) & (data_1_test.age<2000) & ~data_1_test.date_account_created.isnull() ][['age']]
    data_1_test.loc[((data_1_test.age>1000) & (data_1_test.age<2000) & (~data_1_test.date_account_created.isnull())),'age' ]=year_date_account_created['year_date_account_created']-age['age']
    data_1_test.loc[(data_1_test['age']<15) | (data_1_test['age']>95),'age']=np.nan
    data_1_test=data_1_test.drop(['date_first_booking','date_account_created','timestamp_first_active','year_date_account_created','year_date_first_booking'],axis=1)
      #Define age group
    def set_age_group(x):
        if x < 40:return 'Young'
        elif x >=40 and x < 60:return 'Middle'
        elif x >= 60 and x <= 125:return 'Old'
        else:return 'Unknown_age'
    data_1_test['age_group'] = data_1_test['age'].apply(set_age_group)
    #Replace NAN to unknown.
    data_1_test[['gender','signup_method','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser']]=data_1_test[['gender','signup_method' ,'language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser']].fillna('-unknown-')
    data_1_test[['signup_flow','age']]=data_1_test[['signup_flow','age']].fillna(0)
    #weekday_account_created,day_account_created,month_account_created,weekday_first_active,day_first_active,month_first_active
    user_data_test=data_1_test[['id','age','age_group','gender','signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
         'signup_app', 'first_device_type', 'first_browser','weekday_account_created', 'day_account_created',
         'month_account_created', 'year_account_created', 'weekday_first_active','day_first_active', 'month_first_active','year_first_active','time_gap']]
    #Grouping multiple rows of dataframe with same user_id
    session_df_group = session_df.groupby('user_id', as_index=False).agg(lambda x: x.tolist())
    def conv_to_strings(items):
        items = [ str(i) for i in items ]
        items = [ re.sub('nan','',i) for i in items ] 
        items = ','.join(items)
        return items
    def conv_to_strings_unique(items):
        items = [ str(i) for i in items ]
        items = [ re.sub('nan','',i) for i in items ] 
        items = ','.join(set(items))
        return items
    def replace_nan_to_0(items):
        items = [ 0 if math.isnan(i) else i for i in items ] 
        return items
    session_df_group['action_unique_count'] = session_df_group['action'].apply(lambda i : len(np.unique(i)))
    session_df_group['action_type_unique_count'] = session_df_group['action_type'].apply(lambda i : len(np.unique(i)))
    session_df_group['action_detail_unique_count'] = session_df_group['action_detail'].apply(lambda i : len(np.unique(i)))
    session_df_group['device_type_unique_count'] = session_df_group['device_type'].apply(lambda i : len(np.unique(i)))
    session_df_group['action'] = session_df_group['action'].apply(conv_to_strings)
    session_df_group['action_type'] = session_df_group['action_type'].apply(conv_to_strings)
    session_df_group['action_detail'] = session_df_group['action_detail'].apply(conv_to_strings)
    session_df_group['device_type'] = session_df_group['device_type'].apply(conv_to_strings_unique)
    session_df_group['secs_elapsed'] = session_df_group['secs_elapsed'].apply(replace_nan_to_0)
    session_df_group['secs_elapsed_min'] = session_df_group['secs_elapsed'].apply(lambda i : np.min(i))
    session_df_group['secs_elapsed_max'] = session_df_group['secs_elapsed'].apply(lambda i : np.max(i))
    session_df_group['secs_elapsed_mean'] = session_df_group['secs_elapsed'].apply(lambda i : np.mean(i))
    session_df_group['secs_elapsed_median'] = session_df_group['secs_elapsed'].apply(lambda i : np.median(i))
    session_df_group['secs_elapsed_std'] = session_df_group['secs_elapsed'].apply(lambda i : np.std(i))
    session_df_group['secs_elapsed_var'] = session_df_group['secs_elapsed'].apply(lambda i : np.var(i))
    session_df_group['secs_elapsed_skew'] = session_df_group['secs_elapsed'].apply(lambda i : skew(i))
    session_df_group['secs_elapsed_kurtosis'] = session_df_group['secs_elapsed'].apply(lambda i : kurtosis(i))
    session_df_group['secs_elapsed'] = session_df_group['secs_elapsed'].apply(lambda i : np.sum(i))
    final_df_test = data_1_test.merge(session_df_group, left_on='id', right_on='user_id', how='left')
    #Applying Count Vectorizer (BOW and TFIDF)
    #def tokens(x):return x.split(',')
    categorical_columns=['action','action_type','action_detail','device_type','age_group','gender','signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked','signup_app', 'first_device_type', 'first_browser','weekday_account_created', 'day_account_created','month_account_created', 'weekday_first_active','day_first_active', 'month_first_active']
    numerical_column=['action_unique_count','action_type_unique_count','action_detail_unique_count','device_type_unique_count','age','year_account_created','year_first_active','time_gap','secs_elapsed','secs_elapsed_min','secs_elapsed_max','secs_elapsed_mean','secs_elapsed_median','secs_elapsed_std','secs_elapsed_var','secs_elapsed_skew','secs_elapsed_kurtosis'] 
    test=sp.coo_matrix((0,0))
    for i in categorical_columns:
        with open('C:\\Users\\SWADESH\\Downloads\\production\\vectorizer\\cnt_vct_'+i+'.pkl', "rb" ) as f:
            cnt_vct=pkl.load(f)
        f.close()
        categorical_columns_cnt_vct_test=cnt_vct.transform(final_df_test[i].apply(lambda j :str(j)))
        test=sp.hstack((test, categorical_columns_cnt_vct_test))
    final_df_test=sp.hstack((test,sp.csr_matrix(final_df_test[numerical_column])))
    with open('C:\\Users\\SWADESH\\Downloads\\production\\label_encoder\\label_encoder.pkl', "rb" ) as f:
        le=pkl.load(f)
    f.close()
    with open('C:\\Users\\SWADESH\\Downloads\\production\\final_model\\final_model.pkl', "rb" ) as f:
        cat=pkl.load(f)
    f.close()

    #test_df = pd.read_csv('test_users.csv')
    test_id = data_1_test['id'].values
    pred = cat.predict_proba(final_df_test)

    ids = []
    countries = []
    # Taking the 5 classes with highest probabilities
    for i in range(len(test_id)):
        idx = test_id[i]
        ids += [idx] * 5
        countries += le.inverse_transform(np.argsort(pred[i])[::-1][:5]).tolist()
    # Create submission
    submission = pd.DataFrame({"id" : ids,"country" : countries})
    return submission


# In[16]:


from flask import Flask,render_template,request
import pandas as pd

# In[ ]:


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict_output',methods=['GET','post'])
def prediction():
    user_file_name = request.form['user']
    session_file_name = request.form['session']
    user_column=['id', 'date_account_created', 'timestamp_first_active','date_first_booking', 'gender', 'age', 'signup_method', 'signup_flow','language', 'affiliate_channel', 'affiliate_provider','first_affiliate_tracked', 'signup_app', 'first_device_type','first_browser']
    session_column=['user_id', 'action', 'action_type', 'action_detail', 'device_type','secs_elapsed']
    user=pd.read_csv(user_file_name,header=None,names=user_column)
    session=pd.read_csv(session_file_name,header=None,names=session_column)
    result=final_fun_1(user,session)
    result=result.to_html(header="true", table_id="table")
    return result

if __name__ == '__main__':
    app.run(port=8080,debug = True)

