#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   To get the complete dataset from CMIE
@Author      :   siyi.sun
@Time        :   2025/01/23 23:40:15
"""

# export PATH="/home/siyi.sun/miniconda3/bin:$PATH"

import pandas as pd
dtype = {"":,
         "":,
         "":,}
data = pd.read_csv('data_need.csv', index_col=0, low_memory=False)
data.drop(columns = ['HAS_MOBILE','WILL_EMIGRATE'],inplace = True)
