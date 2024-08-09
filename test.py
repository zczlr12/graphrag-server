import pandas as pd

 
# 创建一个DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

df = df[df["B"] == 5]
print(df)
print("---------")
# 将第一行转换为Series
row_as_series = df.iloc[0]  # 通过索引位置
print(row_as_series)