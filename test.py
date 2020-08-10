import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import os
import datetime
from tensorflow.keras import regularizers

x = pd.read_csv('train.csv')
x_test = pd.read_csv('test.csv')
y_a = x['breed_category'].astype(int)
y_b = x['pet_category']
y_a = y_a.to_frame()
y_b = y_b.to_frame()
x = x.drop('pet_category',axis=1)
x = x.drop('breed_category',axis=1)
x = x.drop('pet_id',axis=1)
pet_id = x_test['pet_id']
x_test = x_test.drop('pet_id', axis=1)

# this line converts the string object in Timestamp object
x['issue_date'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in x["issue_date"]]

x_test['issue_date'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in x_test["issue_date"]]

# extracting date from timestamp
x['issue_dates'] = [datetime.datetime.date(d) for d in x['issue_date']]

x_test['issue_dates'] = [datetime.datetime.date(d) for d in x_test['issue_date']]

x = x.drop("issue_date",axis=1)
x_test = x_test.drop("issue_date", axis=1)

x['listing_date'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in x["listing_date"]]
x['listing_dates'] = [datetime.datetime.date(d) for d in x['listing_date']]
x = x.drop("listing_date",axis=1)

x['Difference_dates'] = x['listing_dates'].sub(x['issue_dates'], axis=0)
x['Difference_dates'] = x['Difference_dates'] / np.timedelta64(1, 'D')

x_test['listing_date'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in x_test["listing_date"]]
x_test['listing_dates'] = [datetime.datetime.date(d) for d in x_test['listing_date']]
x_test = x_test.drop("listing_date", axis=1)

x_test['Difference_dates'] = x_test['listing_dates'].sub(x_test['issue_dates'], axis=0)
x_test['Difference_dates'] = x_test['Difference_dates'] / np.timedelta64(1, 'D')
x['listing_dates'] = pd.to_datetime(x['listing_dates'])
x['issue_dates'] = pd.to_datetime(x['issue_dates'])
x_test['listing_dates'] = pd.to_datetime(x_test['listing_dates'])
x_test['issue_dates'] = pd.to_datetime(x_test['issue_dates'])

x['listing_year'] = x['listing_dates'].dt.year
x['listing_month'] = x['listing_dates'].dt.month
x['issue_year'] = x['issue_dates'].dt.year
x['issue_month'] = x['issue_dates'].dt.month
x['listing_day'] = x['listing_dates'].dt.day
x['issue_day'] = x['issue_dates'].dt.day

x_test['listing_year'] = x_test['listing_dates'].dt.year
x_test['listing_month'] = x_test['listing_dates'].dt.month
x_test['issue_year'] = x_test['issue_dates'].dt.year
x_test['issue_month'] = x_test['issue_dates'].dt.month
x_test['listing_day'] = x_test['listing_dates'].dt.day
x_test['issue_day'] = x_test['issue_dates'].dt.day

x = x.drop("listing_dates",axis=1)
x = x.drop("issue_dates",axis=1)
x_test = x_test.drop("listing_dates", axis=1)
x_test = x_test.drop("issue_dates", axis=1)


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

imputer = imputer.fit(x.iloc[:,0:1])
x.iloc[:,0:1] = imputer.transform(x.iloc[:,0:1])
x.iloc[:,0:1] = x.iloc[:,0:1].round()

ordinal_encoder = sklearn.preprocessing.OrdinalEncoder()
x.iloc[:,1:2] = ordinal_encoder.fit_transform(x.iloc[:,1:2])
cat_encoder = sklearn.preprocessing.OneHotEncoder()
cat_1 = x['color_type']
x = x.drop("color_type",axis=1)
cat_1 = cat_1.to_frame()
cat_1 = cat_encoder.fit_transform(cat_1)
cat_1 = pd.DataFrame(cat_1.toarray())
cat_1.drop([0,1,2],axis = 1,inplace=True)
x = pd.concat([x,cat_1],axis=1)


imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

imputer = imputer.fit(x_test.iloc[:, 0:1])
x_test.iloc[:, 0:1] = imputer.transform(x_test.iloc[:, 0:1])
x_test.iloc[:, 0:1] = x_test.iloc[:, 0:1].round()

ordinal_encoder = sklearn.preprocessing.OrdinalEncoder()
x_test.iloc[:, 1:2] = ordinal_encoder.fit_transform(x_test.iloc[:, 1:2])
cat_encoder = sklearn.preprocessing.OneHotEncoder()
cat_1_t = x_test['color_type']
x_test = x_test.drop("color_type", axis=1)
cat_1_t = cat_1_t.to_frame()
cat_1_t = cat_encoder.fit_transform(cat_1_t)
cat_1_t = pd.DataFrame(cat_1_t.toarray())
cat_1_t.drop(0,axis = 1,inplace=True)
x_test = pd.concat([x_test, cat_1_t], axis=1)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
X = StandardScaler()
x = X.fit_transform(x)
x=pd.DataFrame(x)
x_test = X.transform(x_test)
x_test=pd.DataFrame(x_test)


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=1000)
clf.fit(x,y_a)
y_pred_a=clf.predict(x_test)
y_pred_a = pd.DataFrame(y_pred_a)
x = pd.concat([x,y_a],axis=1)
x_test = pd.concat([x_test,y_pred_a], axis=1)

clf=RandomForestClassifier(n_estimators=1000)
clf.fit(x,y_b)
y_pred_b=clf.predict(x_test)

y_pred_a= y_pred_a.to_numpy()
y_pred_a = y_pred_a.reshape((y_pred_a.shape[0]))
y_pred_b= y_pred_b.to_numpy()
y_pred_b = y_pred_b.reshape((y_pred_b.shape[0]))
y_pred_a = pd.Series(y_pred_a,name="breed_category")
y_pred_b = pd.Series(y_pred_b,name="pet_category")
y_pred_a = pd.DataFrame(y_pred_a)
y_pred_b = pd.DataFrame(y_pred_b)
pet_id = pd.Series(pet_id,name="pet_id")
pet_id = pd.DataFrame(pet_id)
submission = pd.concat([pet_id,y_pred_a,y_pred_b],axis = 1)
submission.to_csv("SUBMISSION_120.csv",index=False)