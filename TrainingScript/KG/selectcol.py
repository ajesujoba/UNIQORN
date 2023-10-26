import pandas as pd  

data1 = pd.read_csv("negative_ex.csv")
df = data1[['questionId','question','context','label']]
df = df.drop_duplicates()

df.to_csv(r'posneg_train.csv', index = False)
