import pandas as pd
import random
import csv



loc=str(raw_input("Data source: "))
limit=int(raw_input("Percent (out of 10) items to take: "))

pre_df=pd.read_csv(loc)
data=pre_df.values.tolist()
columns=pre_df.columns.tolist()

new_list=list()

for dp in data:
    x=random.randint(0,9)
    if(x <= limit - 1):
        new_list.append(dp)

with open(str(loc.split('.')[0]) + "_test_data.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(new_list)


