import pandas as pd
import mlxtend
import pyfpgrowth

from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

data = pd.read_csv("GroceryStoreDataSet.csv")
data.index.rename('TID', inplace=True)
data.rename(columns={"MILK,BREAD,BISCUIT": 'item_list'}, inplace=True)

df1 = data['item_list'].str.split(',')
# print(data.mean())
df = pd.DataFrame(df1)
print(df)

#fp = fpgrowth(df, min_support=0.5, use_colnames=True)
#print(fp)


transactions=df

min_support=2
tno=0
tran=[]
for i in df['item_list']:
    tno+=1
    for j in i:
        print(j)
        tran.append(j)

min_threshold=min_support/tno
#min_threshold=0.5

confi=0.5
# support=0.5*4
# print(support)
te = TransactionEncoder()
te_array = te.fit(df).transform(df)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)
frequent_itemsets_fp=fpgrowth(df, min_support=0.02, use_colnames=True)

rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.8)
print(rules_fp)