import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

data = pd.read_csv("GroceryStoreDataSet.csv")
data.index.rename('TID', inplace=True)
data.rename(columns={"MILK,BREAD,BISCUIT": 'item_list'}, inplace=True)

df1 = data['item_list'].str.split(',')
# print(data.mean())
df = pd.DataFrame(df1)
print(df)

fp = fpgrowth(df, min_support=0.5, use_colnames=True)
print(fp)
