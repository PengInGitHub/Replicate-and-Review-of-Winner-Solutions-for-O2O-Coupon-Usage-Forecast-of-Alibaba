#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:06:47 2017

@author: pengchengliu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import date
################################
#       0.datasete info        #
################################
#resource:
#nunique() is more appropriate than value_counts() in this case
#https://stackoverflow.com/questions/15411158/pandas-countdistinct-equivalent

#load data
off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv',header=None)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

off_train = pd.read_csv('ccf_offline_stage1_train.csv',header=None)
off_train.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']

on_train = pd.read_csv('ccf_online_stage1_train.csv',header=None)
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']


#off_train statistics
len(off_train)#1754884, 1755k transcations
unique_user = (off_train.user_id.nunique())#539438, 540K identical users

unique_merchant = (off_train.merchant_id.nunique())#8415, 8.4K identical merchants

have_no_coupon = np.sum(off_train.coupon_id=='null',axis=0)#701602, 701k transcations have no coupon                  

have_coupon = np.sum(off_train.coupon_id!='null',axis=0)#1053282, 1053k transcations have coupon

use_no_coupon = np.sum((off_train.coupon_id!='null')&((off_train.date=='null')),axis=0)#977900, 978k have but given up coupon
                   
use_coupon = np.sum((off_train.coupon_id!='null')&((off_train.date!='null')),axis=0)#75382, 75k use coupon
                   
off_train[off_train.date_received!='null'].date_received.min()#20160101
off_train[off_train.date_received!='null'].date_received.max()#20160615

off_train[off_train.date!='null'].date.min()#'20160101'
off_train[off_train.date!='null'].date.max()#'20160630'         
#summary
#1755k records, 54k users, 8k merchants.
#date of coupon received(01 Jan - 15 Jun)
#date of coupon used(01 Jan - 30 Jun)

#701k have no coupon, 1053k have coupon. Among the guys have coupon, 978 k give up (0) and 75 k use coupon (1)
#unbalance rate 7:93 



#on_train statistics
len(on_train)#11429826, 11430k transcations, 6.5 times more than off_train
unique_user = (on_train.user_id.nunique())#762858, 763K identical users

len(set(off_train.user_id).intersection(on_train.user_id))#267448 of 539438 users in off_train show up in 762858 users in on_train             
              
unique_merchant = (on_train.merchant_id.nunique())#7999, 8K identical merchants

len(set(off_train.merchant_id).intersection(on_train.merchant_id))#no overlapping??           
                  
have_no_coupon = np.sum(on_train.coupon_id=='null',axis=0)#10557k transcations have no coupon                  

have_coupon = np.sum(on_train.coupon_id!='null',axis=0)# 872k transcations have coupon

#8:92 have : have no coupon                    
use_no_coupon = np.sum((on_train.coupon_id!='null')&((on_train.date=='null')),axis=0)#656k have but given up coupon
                   
use_coupon = np.sum((on_train.coupon_id!='null')&((on_train.date!='null')),axis=0)#216k use coupon
#25:75 unbalance rate
                   
on_train[on_train.date_received!='null'].date_received.min()#20160101
on_train[on_train.date_received!='null'].date_received.max()#20160615

on_train[on_train.date!='null'].date.min()#'20160101'
on_train[on_train.date!='null'].date.max()#'20160630'   
 
                  
#comparison of on_train to off_train
#11430 k vs.1755k records, 6.5 times larger
#763k vs. 540k identical users, 267k same users
#8.0k vs. 8.4k identical merchants, no overlapping
#10557k vs. 701k have no coupon
#872k vs. 1053k have coupon
#25:75 vs. 7:93 unbalance rate


#off_test statistics
len(off_test)#113640, 113k transcations, 6% of off_train
unique_user = (off_test.user_id.nunique())#76K identical users

len(set(off_test.user_id).intersection(off_train.user_id))#76307 of 76309 occur in off_train             
              
unique_merchant = (off_test.merchant_id.nunique())#1599 identical merchants

len(set(off_test.merchant_id).intersection(off_train.merchant_id))#1558 0f 1559 merchants occur in both data sets          
 
off_test.date_received.min()#'20160701'
off_test.date_received.max()#'20160731'                  
################################
#       1.sliding window       #
################################

feature1 = off_train[((off_train.date>='20160101')&(off_train.date<='20160413'))|((off_train.date=='null')&(off_train.date_received>='20160101')&(off_train.date_received<='20160413'))]
dataset1 = off_train[(off_train.date_received>='20160414')&(off_train.date_received<='20160514')]

feature2 = off_train[((off_train.date>='20160201')&(off_train.date<='20160514'))|((off_train.date=='null')&(off_train.date_received>='20160201')&(off_train.date_received<='20160514'))]
dataset2 = off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')]

feature3 = off_train[((off_train.date>='20160315')&(off_train.date<='20160630'))|((off_train.date=='null')&(off_train.date_received>='20160315')&(off_train.date_received<='20160630'))]
dataset3 = off_test

#            date_received       shape
#feature1 (20160101,20160413)    995k
#data1    (20160414,20160514)    137k
 
#feature2 (20160201,20160514)    813k
#data2    (20160515,20160615)    258k

#feature3 (20160315,20160630)    1034k
#data3    (20160701,20160731)    114k

############# 2. other feature ##################
"""
5. other feature:
      this_month_user_receive_all_coupon_count
      this_month_user_receive_same_coupon_count
      this_month_user_receive_same_coupon_lastone
      this_month_user_receive_same_coupon_firstone
      this_day_user_receive_all_coupon_count
      this_day_user_receive_same_coupon_count
      day_gap_before, day_gap_after  (receive the same coupon)
"""
          
#for dataset 3
t = dataset3[['user_id']]          
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()


t1 = dataset3[['user_id','coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()


t2 = dataset3[['user_id','coupon_id','date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
#https://stackoverflow.com/questions/27298178/concatenate-strings-from-several-rows-using-pandas-groupby
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
#https://stackoverflow.com/questions/4787479/python-filter-string-only-showing-tags-less-than-2-words
t2 = t2[t2.receive_number>1]
t2['max_date_received'] = t2.date_received.apply(lambda s:max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(lambda s:min([int(d) for d in s.split(':')]))
t2 = t2[['user_id','coupon_id','max_date_received','min_date_received']]


t3 = dataset3[['user_id','coupon_id','date_received']]
t3 = pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received - t3.min_date_received
def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1 #those only receive once
        
t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
t3 = t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]


t4 = dataset3[['user_id','date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id','date_received']).agg('sum').reset_index()


t5 = dataset3[['user_id','coupon_id','date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()


t6 = dataset3[['user_id','coupon_id','date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t6.rename(columns={'date_received':'dates'},inplace=True)


def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
        
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
    

t7 = dataset3[['user_id','coupon_id','date_received']]
t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

other_feature3 = pd.merge(t1,t,on='user_id')
other_feature3 = pd.merge(other_feature3,t3,on=['user_id','coupon_id'])
other_feature3 = pd.merge(other_feature3,t4,on=['user_id','date_received'])
other_feature3 = pd.merge(other_feature3,t5,on=['user_id','coupon_id','date_received'])
other_feature3 = pd.merge(other_feature3,t7,on=['user_id','coupon_id','date_received'])
other_feature3.to_csv('other_feature3.csv',index=None)
print (other_feature3.shape)


#for dataset2
t = dataset2[['user_id']]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()
t.this_month_user_receive_all_coupon_count.describe()


t1 = dataset2[['user_id','coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()


t2 = dataset2[['user_id','coupon_id','date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
t2 = t2[t2.receive_number>1]
t2['max_date_received'] = t2.date_received.apply(lambda s:max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(lambda s:min([int(d) for d in s.split(':')]))
t2 = t2[['user_id','coupon_id','max_date_received','min_date_received']]


t3 = dataset2[['user_id','coupon_id','date_received']]
t3 = pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype('int')
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype('int') - t3.min_date_received
def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1 #those only receive once
        
t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
t3 = t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]


t4 = dataset2[['user_id','date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id','date_received']).agg('sum').reset_index()


t5 = dataset2[['user_id','coupon_id','date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()

t6 = dataset2[['user_id','coupon_id','date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t6.rename(columns={'date_received':'dates'},inplace=True)

def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
        
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
    

t7 = dataset2[['user_id','coupon_id','date_received']]
t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)


t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

other_feature2 = pd.merge(t1,t,on='user_id')
other_feature2 = pd.merge(other_feature2,t3,on=['user_id','coupon_id'])
other_feature2 = pd.merge(other_feature2,t4,on=['user_id','date_received'])
other_feature2 = pd.merge(other_feature2,t5,on=['user_id','coupon_id','date_received'])
other_feature2 = pd.merge(other_feature2,t7,on=['user_id','coupon_id','date_received'])
other_feature2.to_csv('other_feature2.csv',index=None)
print (other_feature2.shape)



#for dataset1
t = dataset1[['user_id']]
t['this_month_user_receive_all_coupon_count'] = 1
t = t.groupby('user_id').agg('sum').reset_index()


t1 = dataset1[['user_id','coupon_id']]
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()


t2 = dataset1[['user_id','coupon_id','date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t2['receive_number'] = t2.date_received.apply(lambda s:len(s.split(':')))
t2 = t2[t2.receive_number>1]
t2['max_date_received'] = t2.date_received.apply(lambda s:max([int(d) for d in s.split(':')]))
t2['min_date_received'] = t2.date_received.apply(lambda s:min([int(d) for d in s.split(':')]))
t2 = t2[['user_id','coupon_id','max_date_received','min_date_received']]


t3 = dataset1[['user_id','coupon_id','date_received']]
t3 = pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
t3['this_month_user_receive_same_coupon_lastone'] = t3.max_date_received - t3.date_received.astype('int')
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype('int') - t3.min_date_received
def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        return -1 #those only receive once
        
t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
t3 = t3[['user_id','coupon_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]


t4 = dataset1[['user_id','date_received']]
t4['this_day_user_receive_all_coupon_count'] = 1
t4 = t4.groupby(['user_id','date_received']).agg('sum').reset_index()


t5 = dataset1[['user_id','coupon_id','date_received']]
t5['this_day_user_receive_same_coupon_count'] = 1
t5 = t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()


t6 = dataset1[['user_id','coupon_id','date_received']]
t6.date_received = t6.date_received.astype('str')
t6 = t6.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
t6.rename(columns={'date_received':'dates'},inplace=True)


def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
        
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (date(int(d[0:4]),int(d[4:6]),int(d[6:8]))-date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
    

t7 = dataset1[['user_id','coupon_id','date_received']]
t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
t7['date_received_date'] = t7.date_received.astype('str') + '-' + t7.dates
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

other_feature1 = pd.merge(t1,t,on='user_id')
other_feature1 = pd.merge(other_feature1,t3,on=['user_id','coupon_id'])
other_feature1 = pd.merge(other_feature1,t4,on=['user_id','date_received'])

other_feature1 = pd.merge(other_feature1,t5,on=['user_id','coupon_id','date_received'])
other_feature1 = pd.merge(other_feature1,t7,on=['user_id','coupon_id','date_received'])
other_feature1.to_csv('other_feature1.csv',index=None)
print (other_feature1.shape)


############# 3. coupon related feature   #############
"""
2.coupon related: 
      discount_rate. discount_man. discount_jian. is_man_jian
      day_of_week,day_of_month. (date_received)
"""
def calc_discount_rate(s):
    s =str(s)
    s = s.split(':')
    if len(s)==1:
        return float(s[0])
    else:
        return 1.0-float(s[1])/float(s[0])

def get_discount_man(s):
    s =str(s)
    s = s.split(':')
    if len(s)==1:
        return 'null'
    else:
        return int(s[0])
        
def get_discount_jian(s):
    s =str(s)
    s = s.split(':')
    if len(s)==1:
        return 'null'
    else:
        return int(s[1])

def is_man_jian(s):
    s =str(s)
    s = s.split(':')
    if len(s)==1:
        return 0
    else:
        return 1



#dataset3
dataset3['day_of_week'] = dataset3.date_received.astype('str').apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
#apply(lambda x:x**2)
#date: apply(lambda x:date(int(x[0:4]),int(4:6),int(6:8)))
#date to weekday: .weekday()+1
                          
dataset3['day_of_month'] = dataset3.date_received.astype('str').apply(lambda x:int(x[6:8]))#retrive day from date

#take difference of two days
dataset3['days_distance'] = dataset3.date_received.astype('str').apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,6,30)).days)

#retrive string ahead of semicolon, which is the 'purchase more than' in 'rebate of ... for purchase more than ...'
#for example, give you 20 euro rebate for a purchase amounts more than 100 euro
#value of rebate : value of purchase are exactly the two parameters here, seperated by :
dataset3['discount_man'] = dataset3.discount_rate.apply(get_discount_man)

#return value of how much money is returned as rebate
dataset3['discount_jian'] = dataset3.discount_rate.apply(get_discount_jian)

#return binary values of the type of discount is 'rebate of ... if purchase amounts more than ...', i.e. if the value contains ':'
dataset3['is_man_jian'] = dataset3.discount_rate.apply(is_man_jian)

#replace '... : ...' type of 'discount_rate' by discount_rate
dataset3['discount_rate'] = dataset3.discount_rate.apply(calc_discount_rate)

d = dataset3[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()#frequency of counpon occurs
dataset3 = pd.merge(dataset3,d,on='coupon_id',how='left')
dataset3.to_csv('coupon3_feature.csv',index=None)


#dataset2
dataset2['day_of_week'] = dataset2.date_received.astype('str').apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
dataset2['day_of_month'] = dataset2.date_received.astype('str').apply(lambda x:int(x[6:8]))
dataset2['days_distance'] = dataset2.date_received.astype('str').apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,5,14)).days)
dataset2['discount_man'] = dataset2.discount_rate.apply(get_discount_man)
dataset2['discount_jian'] = dataset2.discount_rate.apply(get_discount_jian)
dataset2['is_man_jian'] = dataset2.discount_rate.apply(is_man_jian)
dataset2['discount_rate'] = dataset2.discount_rate.apply(calc_discount_rate)
d = dataset2[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()
dataset2 = pd.merge(dataset2,d,on='coupon_id',how='left')
dataset2.to_csv('coupon2_feature.csv',index=None)

#dataset1
dataset1['day_of_week'] = dataset1.date_received.astype('str').apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
dataset1['day_of_month'] = dataset1.date_received.astype('str').apply(lambda x:int(x[6:8]))
dataset1['days_distance'] = dataset1.date_received.astype('str').apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,4,13)).days)
dataset1['discount_man'] = dataset1.discount_rate.apply(get_discount_man)
dataset1['discount_jian'] = dataset1.discount_rate.apply(get_discount_jian)
dataset1['is_man_jian'] = dataset1.discount_rate.apply(is_man_jian)
dataset1['discount_rate'] = dataset1.discount_rate.apply(calc_discount_rate)
d = dataset1[['coupon_id']]
d['coupon_count'] = 1
d = d.groupby('coupon_id').agg('sum').reset_index()
dataset1 = pd.merge(dataset1,d,on='coupon_id',how='left')
dataset1.to_csv('coupon1_feature.csv',index=None)


############# 4.merchant related feature   #############
"""
1.merchant related: 
      total_sales. sales_use_coupon.  total_coupon
      coupon_rate = sales_use_coupon/total_sales.  
      transfer_rate = sales_use_coupon/total_coupon. 
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon
"""

#for dataset3
merchant3 = feature3[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant3[['merchant_id']]
t.drop_duplicates(inplace=True)#1037k to 8k, identical merchant

t1 = merchant3[merchant3.date!='null'][['merchant_id']]#all transcations 
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()#aggregrated transcations by merchant

t2 = merchant3[(merchant3.date!='null')&(merchant3.coupon_id!='null')][['merchant_id']]#all transcations via goupon
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()#all transcations via goupon by merchant

t3 = merchant3[merchant3.coupon_id!='null'][['merchant_id']]#all records with coupon
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()#total coupons

t4 = merchant3[(merchant3.date!='null')&(merchant3.coupon_id!='null')][['merchant_id','distance']]#transcations with coupon
t4.replace('null',-1,inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1,np.nan,inplace=True)#t4 contains the distance where transcations occured via coupons
#replace(a value, as nan, inplace=True)

t5 = t4.groupby('merchant_id').agg('min').reset_index()#record the min distance by merchant that has transcations conducted via coupon
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()#record the max distance by merchant that has transcations conducted via coupon
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()#record the average distance by merchant that has transcations conducted via coupon
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()#record the median distance by merchant that has transcations conducted via coupon
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)

merchant3_feature = pd.merge(t,t1,on='merchant_id',how='left')#merchant has transcations, aggregrated transcations by merchant
merchant3_feature = pd.merge(merchant3_feature,t2,on='merchant_id',how='left')#plus count of transcations via coupon
merchant3_feature = pd.merge(merchant3_feature,t3,on='merchant_id',how='left')#plus all coupons of the merchant
merchant3_feature = pd.merge(merchant3_feature,t5,on='merchant_id',how='left')#plus min distance, 0 to 10 
merchant3_feature = pd.merge(merchant3_feature,t6,on='merchant_id',how='left')#plus max distance, 0 to 10 
merchant3_feature = pd.merge(merchant3_feature,t7,on='merchant_id',how='left')#plus mean distance, 0 to 10 
merchant3_feature = pd.merge(merchant3_feature,t8,on='merchant_id',how='left')#plus median distance, 0 to 10 

np.sum(merchant3_feature.sales_use_coupon.isnull(),axis=0)#4429 of 8239 shops have no transcations completed via coupon                            
merchant3_feature.sales_use_coupon = merchant3_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
#data.column_1=data.column_1.replace(np.nan,0) replace nan by a specific value
                                                                               
merchant3_feature['merchant_coupon_transfer_rate'] = merchant3_feature.sales_use_coupon.astype('float') / merchant3_feature.total_coupon
#this rate is critical for coupon usage preduction                 
                 
merchant3_feature['coupon_rate'] = merchant3_feature.sales_use_coupon.astype('float') / merchant3_feature.total_sales
#coupon_rate describes how much a shop is dependent on coupon for the given period                  

merchant3_feature.total_coupon = merchant3_feature.total_coupon.replace(np.nan,0) #fillna with 0
merchant3_feature.to_csv('merchant3_feature.csv',index=None)


#for dataset2
merchant2 = feature2[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant2[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant2[merchant2.date!='null'][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()#agg occured orders

t2 = merchant2[(merchant2.date!='null')&(merchant2.coupon_id!='null')][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()#agg occured orders via coupon

t3 = merchant2[merchant2.coupon_id!='null'][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()#all coupons of the shop, no matter they are used or not

t4 = merchant2[(merchant2.date!='null')&(merchant2.coupon_id!='null')][['merchant_id','distance']]
t4.replace('null',-1,inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1,np.nan,inplace=True)

t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)

merchant2_feature = pd.merge(t,t1,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t2,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t3,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t5,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t6,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t7,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t8,on='merchant_id',how='left')

merchant2_feature.sales_use_coupon = merchant2_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
merchant2_feature['merchant_coupon_transfer_rate'] = merchant2_feature.sales_use_coupon.astype('float') / merchant2_feature.total_coupon

merchant2_feature['coupon_rate'] = merchant2_feature.sales_use_coupon.astype('float') / merchant2_feature.total_sales
merchant2_feature.total_coupon = merchant2_feature.total_coupon.replace(np.nan,0) #fillna with 0
merchant2_feature.to_csv('merchant2_feature.csv',index=None)

#for dataset1
merchant1 = feature1[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant1[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant1[merchant1.date!='null'][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant1[(merchant1.date!='null')&(merchant1.coupon_id!='null')][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant1[merchant1.coupon_id!='null'][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant1[(merchant1.date!='null')&(merchant1.coupon_id!='null')][['merchant_id','distance']]
t4.replace('null',-1,inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1,np.nan,inplace=True)

t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)


merchant1_feature = pd.merge(t,t1,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t2,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t3,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t5,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t6,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t7,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t8,on='merchant_id',how='left')

merchant1_feature.sales_use_coupon = merchant1_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
merchant1_feature['merchant_coupon_transfer_rate'] = merchant1_feature.sales_use_coupon.astype('float') / merchant1_feature.total_coupon
merchant1_feature['coupon_rate'] = merchant1_feature.sales_use_coupon.astype('float') / merchant1_feature.total_sales

merchant1_feature.total_coupon = merchant1_feature.total_coupon.replace(np.nan,0) #fillna with 0
merchant1_feature.to_csv('merchant1_feature.csv',index=None)



############# 5.user related feature   #############
"""
3.user related: 
      count_merchant. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      buy_use_coupon/buy_total
      user_date_datereceived_gap
      
"""

def get_user_date_datereceived_gap(s):
    s = s.split(':')
    return (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8])) - date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days

#for dataset3
user3 = feature3[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

t = user3[['user_id']]
t.drop_duplicates(inplace=True)#351k identical users

t1 = user3[user3.date!='null'][['user_id','merchant_id']]
t1.drop_duplicates(inplace=True)#230 k identical users have transcations
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)#how many shops the user has been to
#data.rename(columns={'name of column','new name of the column'},inplace=True)

t2 = user3[(user3.date!='null')&(user3.coupon_id!='null')][['user_id','distance']]
t2.replace('null',-1,inplace=True)
#data.replace('old value',new value,inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1,np.nan,inplace=True)

t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance':'user_min_distance'},inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance':'user_max_distance'},inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance':'user_median_distance'},inplace=True)

t7 = user3[(user3.date!='null')&(user3.coupon_id!='null')][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()#count of transcations via coupon

t8 = user3[user3.date!='null'][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()#count of transcations

t9 = user3[user3.coupon_id!='null'][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()#count of coupon received

t10 = user3[(user3.date_received!='null')&(user3.date!='null')][['user_id','date_received','date']]#users that use coupons
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id','user_date_datereceived_gap']]#how many days it takes for the user to use the coupon he received

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)


user3_feature = pd.merge(t,t1,on='user_id',how='left')#identical users and numbers of shops have been to
user3_feature = pd.merge(user3_feature,t3,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t4,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t5,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t6,on='user_id',how='left')#user's distance to shops he/she uses coupons

user3_feature = pd.merge(user3_feature,t7,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t8,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t9,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t11,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t12,on='user_id',how='left')
user3_feature = pd.merge(user3_feature,t13,on='user_id',how='left')

user3_feature.count_merchant = user3_feature.count_merchant.replace(np.nan,0)
user3_feature.buy_use_coupon = user3_feature.buy_use_coupon.replace(np.nan,0)
#data.column = data.column.replace(np.nan,0)

user3_feature['buy_use_coupon_rate'] = user3_feature.buy_use_coupon.astype('float') / user3_feature.buy_total.astype('float')
user3_feature['user_coupon_transfer_rate'] = user3_feature.buy_use_coupon.astype('float') / user3_feature.coupon_received.astype('float')
#rate of transcations completed via coupon
#coupon's usage rate by user

user3_feature.buy_total = user3_feature.buy_total.replace(np.nan,0)
user3_feature.coupon_received = user3_feature.coupon_received.replace(np.nan,0)
user3_feature.to_csv('user3_feature.csv',index=None)


#for dataset2
user2 = feature2[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

t = user2[['user_id']]
t.drop_duplicates(inplace=True)
#data.drop_duplicates(inplace=True)

t1 = user2[user2.date!='null'][['user_id','merchant_id']]
t1.drop_duplicates(inplace=True)#different transcations of each user
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()#count of transcatiosn in different stores for each user
t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)

t2 = user2[(user2.date!='null')&(user2.coupon_id!='null')][['user_id','distance']]
t2.replace('null',-1,inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1,np.nan,inplace=True)

t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance':'user_min_distance'},inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance':'user_max_distance'},inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance':'user_median_distance'},inplace=True)

t7 = user2[(user2.date!='null')&(user2.coupon_id!='null')][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()#all orders via coupon

t8 = user2[user2.date!='null'][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()#all orders

t9 = user2[user2.coupon_id!='null'][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()#all coupons

t10 = user2[(user2.date_received!='null')&(user2.date!='null')][['user_id','date_received','date']]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id','user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)

user2_feature = pd.merge(t,t1,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t3,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t4,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t5,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t6,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t7,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t8,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t9,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t11,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t12,on='user_id',how='left')
user2_feature = pd.merge(user2_feature,t13,on='user_id',how='left')

user2_feature.count_merchant = user2_feature.count_merchant.replace(np.nan,0)
user2_feature.buy_use_coupon = user2_feature.buy_use_coupon.replace(np.nan,0)

user2_feature['buy_use_coupon_rate'] = user2_feature.buy_use_coupon.astype('float') / user2_feature.buy_total.astype('float')
user2_feature['user_coupon_transfer_rate'] = user2_feature.buy_use_coupon.astype('float') / user2_feature.coupon_received.astype('float')
#coupon dependent rate and coupon usage rate

user2_feature.buy_total = user2_feature.buy_total.replace(np.nan,0)
user2_feature.coupon_received = user2_feature.coupon_received.replace(np.nan,0)
user2_feature.to_csv('user2_feature.csv',index=None)


#for dataset1
user1 = feature1[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

t = user1[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user1[user1.date!='null'][['user_id','merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)

t2 = user1[(user1.date!='null')&(user1.coupon_id!='null')][['user_id','distance']]
t2.replace('null',-1,inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1,np.nan,inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance':'user_min_distance'},inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance':'user_max_distance'},inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance':'user_median_distance'},inplace=True)

t7 = user1[(user1.date!='null')&(user1.coupon_id!='null')][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user1[user1.date!='null'][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user1[user1.coupon_id!='null'][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user1[(user1.date_received!='null')&(user1.date!='null')][['user_id','date_received','date']]
t10['user_date_datereceived_gap'] = t10.date + ':' + t10.date_received
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id','user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)

user1_feature = pd.merge(t,t1,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t3,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t4,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t5,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t6,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t7,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t8,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t9,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t11,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t12,on='user_id',how='left')
user1_feature = pd.merge(user1_feature,t13,on='user_id',how='left')
user1_feature.count_merchant = user1_feature.count_merchant.replace(np.nan,0)
user1_feature.buy_use_coupon = user1_feature.buy_use_coupon.replace(np.nan,0)
user1_feature['buy_use_coupon_rate'] = user1_feature.buy_use_coupon.astype('float') / user1_feature.buy_total.astype('float')
user1_feature['user_coupon_transfer_rate'] = user1_feature.buy_use_coupon.astype('float') / user1_feature.coupon_received.astype('float')
user1_feature.buy_total = user1_feature.buy_total.replace(np.nan,0)
user1_feature.coupon_received = user1_feature.coupon_received.replace(np.nan,0)

user1_feature.to_csv('user1_feature.csv',index=None)



"""
6.user_merchant: times_user_buy_merchant_before. 
"""

#for dataset3
all_user_merchant = feature3[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature3[['user_id','merchant_id','date']]
t = t[t.date!='null'][['user_id','merchant_id']]#records with transcations only
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)#user-merchant pair, count of times the user went to the shop

t1 = feature3[['user_id','merchant_id','coupon_id']]
t1 = t1[t1.coupon_id!='null'][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)#user-merchant pair, count of coupons the user received from the shop

t2 = feature3[['user_id','merchant_id','date','date_received']]
t2 = t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)#user-merchant pair, count of coupons the user used in the shop

t3 = feature3[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature3[['user_id','merchant_id','date','coupon_id']]
t4 = t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)#user-merchant pair, count of transcations without coupons in the shop

user_merchant3 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t1,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t2,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t3,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t4,on=['user_id','merchant_id'],how='left')

user_merchant3.user_merchant_buy_use_coupon = user_merchant3.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant3.user_merchant_buy_common = user_merchant3.user_merchant_buy_common.replace(np.nan,0)

user_merchant3['user_merchant_coupon_transfer_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype('float') / user_merchant3.user_merchant_received.astype('float')
#rate of the number of a user shoped in a given store via coupon to the number of all coupons he/she received from the store
user_merchant3['user_merchant_coupon_buy_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype('float') / user_merchant3.user_merchant_buy_total.astype('float')
#rate of the number of a user shoped in a given store via coupon to the number of times that the user went to the shop

user_merchant3['user_merchant_rate'] = user_merchant3.user_merchant_buy_total.astype('float') / user_merchant3.user_merchant_any.astype('float')
#rate of the number of all items a user purchase in a given shop to the number of the user leaves records in the shop, no matter he purchased or not
user_merchant3['user_merchant_common_buy_rate'] = user_merchant3.user_merchant_buy_common.astype('float') / user_merchant3.user_merchant_buy_total.astype('float')
#rate of the number of a user shoped in a given store not via coupon to the number of times that the user went to the shop

user_merchant3.to_csv('user_merchant3.csv',index=None)

#for dataset2
all_user_merchant = feature2[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)#424k identical user-merchant pairs

t = feature2[['user_id','merchant_id','date']]
t = t[t.date!='null'][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)#all purchase in a given store by user

t1 = feature2[['user_id','merchant_id','coupon_id']]
t1 = t1[t1.coupon_id!='null'][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)#all coupons the user received from the shop

t2 = feature2[['user_id','merchant_id','date','date_received']]
t2 = t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)#all purchase via coupon in the given store by user

t3 = feature2[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)#all records in the given store by user

t4 = feature2[['user_id','merchant_id','date','coupon_id']]
t4 = t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)#all purchase not via coupon in the given store by user

user_merchant2 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t1,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t2,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t3,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t4,on=['user_id','merchant_id'],how='left')

user_merchant2.user_merchant_buy_use_coupon = user_merchant2.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant2.user_merchant_buy_common = user_merchant2.user_merchant_buy_common.replace(np.nan,0)

user_merchant2['user_merchant_coupon_transfer_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype('float') / user_merchant2.user_merchant_received.astype('float')
#rate of the received coupon is used in the given store by user
user_merchant2['user_merchant_coupon_buy_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype('float') / user_merchant2.user_merchant_buy_total.astype('float')
#rate of the transcation in the given store is via coupon by user
user_merchant2['user_merchant_rate'] = user_merchant2.user_merchant_buy_total.astype('float') / user_merchant2.user_merchant_any.astype('float')
#rate of the numeber of transcations in the given store to all the user-merchat records by user
user_merchant2['user_merchant_common_buy_rate'] = user_merchant2.user_merchant_buy_common.astype('float') / user_merchant2.user_merchant_buy_total.astype('float')
#rate of the transcation in the given store is not via coupon by user

user_merchant2.to_csv('user_merchant2.csv',index=None)

#for dataset1
all_user_merchant = feature1[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature1[['user_id','merchant_id','date']]
t = t[t.date!='null'][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature1[['user_id','merchant_id','coupon_id']]
t1 = t1[t1.coupon_id!='null'][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature1[['user_id','merchant_id','date','date_received']]
t2 = t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature1[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature1[['user_id','merchant_id','date','coupon_id']]
t4 = t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant1 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t1,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t2,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t3,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t4,on=['user_id','merchant_id'],how='left')
user_merchant1.user_merchant_buy_use_coupon = user_merchant1.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant1.user_merchant_buy_common = user_merchant1.user_merchant_buy_common.replace(np.nan,0)
user_merchant1['user_merchant_coupon_transfer_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype('float') / user_merchant1.user_merchant_received.astype('float')
user_merchant1['user_merchant_coupon_buy_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype('float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1['user_merchant_rate'] = user_merchant1.user_merchant_buy_total.astype('float') / user_merchant1.user_merchant_any.astype('float')
user_merchant1['user_merchant_common_buy_rate'] = user_merchant1.user_merchant_buy_common.astype('float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1.to_csv('user_merchant1.csv',index=None)
  

##################  generate training and testing set ################
def get_label(s):
    s = s.split(':')
    if s[0]=='null':
        return 0
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return -1
#return lables
#if transcation date is 'null', return 0
#if difference between transcation date and coupon received date is smaller than and equals to 15, return 1
#if difference between transcation date and coupon received date is larger than 15, return -1


#dataset3

coupon3 = pd.read_csv('coupon3_feature.csv')
merchant3 = pd.read_csv('merchant3_feature.csv')
user3 = pd.read_csv('user3_feature.csv')
user_merchant3 = pd.read_csv('user_merchant3.csv')
other_feature3 = pd.read_csv('other_feature3.csv')

dataset3 = pd.merge(coupon3,merchant3,on='merchant_id',how='left')
dataset3 = pd.merge(dataset3,user3,on='user_id',how='left')
dataset3 = pd.merge(dataset3,user_merchant3,on=['user_id','merchant_id'],how='left')
dataset3 = pd.merge(dataset3,other_feature3,on=['user_id','coupon_id','date_received'],how='left')
dataset3.drop_duplicates(inplace=True)
print (dataset3.shape)#(112803, 52)

dataset3.user_merchant_buy_total = dataset3.user_merchant_buy_total.replace(np.nan,0)
dataset3.user_merchant_any = dataset3.user_merchant_any.replace(np.nan,0)
dataset3.user_merchant_received = dataset3.user_merchant_received.replace(np.nan,0)
dataset3['is_weekend'] = dataset3.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
#generate a binary column that is dependent on the values of other columns
#data.new_column=data.old_column.apply(lambda x:value_by_default if x in (range) else value_not_by_default)

weekday_dummies = pd.get_dummies(dataset3.day_of_week)
#dummy: pd.get_dummies(data.column)

weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
#weekday_dummies.shape[1]=7
#data.shape[1] returns the number of columns
                      
dataset3 = pd.concat([dataset3,weekday_dummies],axis=1)
#pd.concat([dataset_1,dataset_2],axis=1)

dataset3.drop(['merchant_id','day_of_week','coupon_count'],axis=1,inplace=True)
dataset3 = dataset3.replace('null',np.nan)
#data = data.replace(value_to_replace,new_value)

dataset3.to_csv('dataset3.csv',index=None)

#dataset 2

coupon2 = pd.read_csv('coupon2_feature.csv')
merchant2 = pd.read_csv('merchant2_feature.csv')
user2 = pd.read_csv('user2_feature.csv')
user_merchant2 = pd.read_csv('user_merchant2.csv')
other_feature2 = pd.read_csv('other_feature2.csv')

dataset2 = pd.merge(coupon2,merchant2,on='merchant_id',how='left')
dataset2 = pd.merge(dataset2,user2,on='user_id',how='left')
dataset2 = pd.merge(dataset2,user_merchant2,on=['user_id','merchant_id'],how='left')
dataset2 = pd.merge(dataset2,other_feature2,on=['user_id','coupon_id','date_received'],how='left')
dataset2.drop_duplicates(inplace=True)
print (dataset2.shape)

dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan,0)
dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan,0)
dataset2.user_merchant_received = dataset2.user_merchant_received.replace(np.nan,0)

dataset2['is_weekend'] = dataset2.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset2.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset2 = pd.concat([dataset2,weekday_dummies],axis=1)

dataset2['label'] = dataset2.date.astype('str') + ':' +  dataset2.date_received.astype('str')#order data and coupon received date
dataset2.label = dataset2.label.apply(get_label)
dataset2.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)

dataset2 = dataset2.replace('null',np.nan)
dataset2.to_csv('dataset2.csv',index=None)

dataset3.shape

#dataset 1

coupon1 = pd.read_csv('coupon1_feature.csv')
merchant1 = pd.read_csv('merchant1_feature.csv')
user1 = pd.read_csv('user1_feature.csv')
user_merchant1 = pd.read_csv('user_merchant1.csv')
other_feature1 = pd.read_csv('other_feature1.csv')
dataset1 = pd.merge(coupon1,merchant1,on='merchant_id',how='left')
dataset1 = pd.merge(dataset1,user1,on='user_id',how='left')
dataset1 = pd.merge(dataset1,user_merchant1,on=['user_id','merchant_id'],how='left')
dataset1 = pd.merge(dataset1,other_feature1,on=['user_id','coupon_id','date_received'],how='left')
dataset1.drop_duplicates(inplace=True)
print (dataset1.shape)

dataset1.user_merchant_buy_total = dataset1.user_merchant_buy_total.replace(np.nan,0)
dataset1.user_merchant_any = dataset1.user_merchant_any.replace(np.nan,0)
dataset1.user_merchant_received = dataset1.user_merchant_received.replace(np.nan,0)
dataset1['is_weekend'] = dataset1.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset1.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset1 = pd.concat([dataset1,weekday_dummies],axis=1)
dataset1['label'] = dataset1.date.astype('str') + ':' +  dataset1.date_received.astype('str')
dataset1.label = dataset1.label.apply(get_label)
dataset1.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset1 = dataset1.replace('null',np.nan)
dataset1.to_csv('dataset1.csv',index=None)

