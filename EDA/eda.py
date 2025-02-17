#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   Exploratory Data Analysis for CMIE
@Author      :   siyi.sun
@Time        :   2024/12/02 22:39:44
"""

# export PATH="/home/siyi.sun/miniconda3/bin:$PATH"

import mysql.connector
import os
import pandas as pd
import numpy as np

db_params = {"host": "data.mayin.org", 
             "user": os.getenv("HHD_AND_FLD_USER"),
             "password":os.getenv("HHD_AND_FLD_PASSWORD"),
             "database":"hhd"}

cnx = mysql.connector.connect(**db_params)
cursor = cnx.cursor()
temp_tab = ("SELECT WAVE_NO,HH_ID,MEM_ID,STATE,HR,REGION_TYPE,MEM_STATUS,AGE_YRS,GENDER,RELIGION,EMPLOYMENT_STATUS,PLACE_OF_WORK,TS_ON_WORK_FOR_EMPLOYER,TS_ON_TRAVEL,TS_ON_OUTDOOR_SPORTS,IS_HOSPITALISED, WAS_HOSPITALISED, HAS_BANK_AC,HAS_MOBILE,IS_HEALTHY FROM people_of_india_wave")
cursor.execute(temp_tab)
col_name = ["WAVE_NO","HH_ID","MEM_ID","STATE","HR","REGION_TYPE","MEM_STATUS","AGE_YRS","GENDER","RELIGION","EMPLOYMENT_STATUS","PLACE_OF_WORK","TS_ON_WORK_FOR_EMPLOYER","TS_ON_TRAVEL","TS_ON_OUTDOOR_SPORTS","IS_HOSPITALISED","WAS_HOSPITALISED","HAS_BANK_AC","HAS_MOBILE","IS_HEALTHY"]
table_use = pd.DataFrame(cursor.fetchall(),columns = col_name)
# table_des = table_use.describe()
# table_des.to_csv('data_describe.csv')
# table_use.shape # 26832063
table_use.to_csv('data_need.csv')
cnx.close()

########################### GET COMPLETE DATASET ########################
##  Replace all values represent missing values with nan or null
table_all_nan = table_use.replace([-100,-99,'Data Not Available'],np.nan)
table_all_nan.isnull().sum()

data_wo_1col = table_all_nan.drop(['WAS_HOSPITALISED'], axis=1)
data_wo_na = data_wo_1col.dropna()
data_wo_na.shape # 5798275

# kickout outliers
data_wo_na_postive = data_wo_na[(data_wo_na['TS_ON_OUTDOOR_SPORTS']>=0)&(data_wo_na['TS_ON_TRAVEL']>=0)&(data_wo_na['TS_ON_WORK_FOR_EMPLOYER']>=0)]
# 1809618
# data_wo_na[data_wo_na['TS_ON_OUTDOOR_SPORTS']<0]['TS_ON_OUTDOOR_SPORTS'] = 0 # 387906 and wave from 18-30
# data_wo_na[data_wo_na['TS_ON_TRAVEL']<0] = 0 # 387906
# data_wo_na[data_wo_na['TS_ON_WORK_FOR_EMPLOYER']<0] = 0 # 3988657

table_query = ("show tables;")
res = cursor.execute(table_query)
cursor.fetchall()
#[('aspirational_wave',), ('hh_char',), ('hh_consumption_monthly',), 
# ('hh_consumption_weekly',), ('hh_income_monthly',), ('member_income_monthly',),
# ('people_of_india_wave',)]

# get household information from hh_income_monthly
query = ("SELECT HH_ID, MONTH, TOT_INC,INC_OF_HH_FRM_ALL_SRCS,INC_OF_ALL_MEMS_FRM_ALL_SRCS from hh_income_monthly")
cursor.execute(query)
col_name = ["HH_ID","MONTH","TOT_INC","INC_OF_HH_FRM_ALL_SRCS","INC_OF_ALL_MEMS_FRM_ALL_SRCS"]
income_use = pd.DataFrame(cursor.fetchall(),columns = col_name)

# get member information from hh_income_monthly
query = ("SELECT HH_ID, MEM_ID, MONTH, INC_OF_MEM_FRM_ALL_SRCS from member_income_monthly")
cursor.execute(query)
col_name = ["HH_ID","MEM_ID","MONTH","INC_OF_MEM_FRM_ALL_SRCS"]
member_income_use = pd.DataFrame(cursor.fetchall(),columns = col_name)

# get member information from hh_income_monthly
query = ("SELECT * from member_income_monthly where HH_ID = 48607646 and MEM_ID = 1")
cursor.execute(query)
col_name = ["HH_ID","MEM_ID","MONTH","INC_OF_MEM_FRM_ALL_SRCS"]
member_income_use = pd.DataFrame(cursor.fetchall(),columns = col_name)


# # check columns
# query = ("SHOW COLUMNS FROM hh_income_monthly")
# cursor.execute(query)
# list_col = pd.DataFrame(cursor.fetchall())


