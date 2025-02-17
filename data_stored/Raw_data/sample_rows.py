#!/home/siyi.sun/miniconda3/bin python3
# -*- coding: UTF-8 -*-
"""
@Description :   sample from a large csv file
@Author      :   siyi.sun
@Time        :   2025/02/02 02:37:14
"""

import pandas as pd
import random
import argparse

parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--file','-f',type=str, required=True, help="The absolute path of csv file to sample.")
args = parser.parse_args()

p = 0.01  # 1% of the lines
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
df = pd.read_csv(
         args.file,
         header=0, 
         index_col=0,
         skiprows=lambda i: i>0 and random.random() > p
)
df.to_csv("sample_"+args.file)