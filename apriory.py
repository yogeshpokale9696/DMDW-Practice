import pandas as pd
from apyori import apriori
from tabulate import tabulate

data = pd.read_csv("GroceryStoreDataSet.csv")

print(data.columns)
print(data.dropna())

no_cols = len(data.columns)
print("columns", no_cols)
no_rows = len(data)
print("rows", no_rows)

records = []
for i in range(0, no_rows):
    records.append([str(data.values[i, j]) for j in range(0, no_cols)])
    # print("records are :\n",records)
association_rules = apriori(records, min_support=0.08, min_confidence=0.02, min_lift=1, min_length=2)
association_results = list(association_rules)
df = pd.DataFrame(association_results)
print(type(association_results))
print(len(association_results))
print(association_results)
print(tabulate(association_results))

"""
for item in association_rules:
    pair = item[0]
    items = [x for x in pair]
    print("Rules:" + items[0] + "->" + items[1])
    print("support:" + str(item[1]))
    print("confidense:" + str(item[2][0][2]))
    print("lift:" + str(item[2][0][3]))
    print("===================================")
path = "C:/Users/USER/Desktop/MSC (C.S) Part 1/Shreyas Joshi/Apriory"
os.chdir(path)
wb=Workbook()
wb.new_sheet("OutPut data", data=association_results)
wb.save(path+"/OutPut data.xlsx")
"""
