import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

data = pd.read_csv("GroceryStoreDataSet.csv", names=['products'], header=None)

print(data.columns)
print(data.dropna)

no_cols = len(data.columns)
no_rows = len(data)
print("No of columns:", no_cols)
print("No of rows:", no_rows)
data = list(data["products"].apply(lambda x: x.split(',')))
print(":\n", data)

te = TransactionEncoder()
te_data = te.fit(data).transform(data)

df = pd.DataFrame(te_data, columns=te.columns_)

fp1 = fpgrowth(df, min_support=0.1, use_colnames=True)
print(fp1)
print(fp1.sort_values(by="support", ascending=False))

pd.set_option('display.max_columns', 10)
rules1 = association_rules(fp1)
print(rules1)
