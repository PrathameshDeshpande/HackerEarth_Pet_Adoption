import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import os
import datetime

x = pd.read_csv('train.csv')
y = x['pet_category']
y = y.to_frame()
x = x.drop('pet_category',axis=1)
x = x.drop('pet_id',axis=1)

# this line converts the string object in Timestamp object
x['issue_date'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in x["issue_date"]]

# extracting date from timestamp
x['issue_dates'] = [datetime.datetime.date(d) for d in x['issue_date']]

x = x.drop("issue_date",axis=1)

x['listing_date'] = [datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in x["listing_date"]]
x['listing_dates'] = [datetime.datetime.date(d) for d in x['listing_date']]
x = x.drop("listing_date",axis=1)

x['Difference_dates'] = x['listing_dates'].sub(x['issue_dates'], axis=0)
x['Difference_dates'] = x['Difference_dates'] / np.timedelta64(1, 'D')
x = x.drop("listing_dates",axis=1)
x = x.drop("issue_dates",axis=1)

