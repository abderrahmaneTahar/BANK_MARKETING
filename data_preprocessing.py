# %%
import pandas as pd
df = pd.read_csv("../banks.csv")

# %%
df["default"]=df["default"].replace(to_replace=['no', 'yes'], value=[0, 1])
df["housing"]=df["housing"].replace(to_replace=['no', 'yes'], value=[0, 1])
df["loan"]=df["loan"].replace(to_replace=['no', 'yes'], value=[0, 1])
df["y"]=df["y"].replace(to_replace=['no', 'yes'], value=[0, 1])

df["month"]=df["month"].replace(to_replace=['oct', 'may', 'apr', 'jun', 'feb', 'aug', 'jan', 'jul', 'nov','sep', 'mar', 'dec'], value=[10,5,4,6,2,8,1,7,11,9,3,12])
# %%
df=pd.get_dummies(data=df, columns=["job","marital","contact","poutcome"])
# %%
df["education"] = df["education"].replace(to_replace=["unknown"],value=["secondary"])
df["education"] = df["education"].replace(to_replace=["primary","secondary","tertiary"],value=[1,2,3])
df.education.value_counts()
df.head()

