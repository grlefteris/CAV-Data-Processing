import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

path='/Users/lefteris/Desktop/untitled1/jrc_athens/results/9200_10200_data.xls'

df = pd.read_excel(path, sheetname='Sheet 1')

print("Column headings:")
print(df.columns)