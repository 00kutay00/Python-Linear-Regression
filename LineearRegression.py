from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("audi.csv")

df = df.drop(columns=["index", "href", "PPY", "MileageRank", "PriceRank", "PPYRank", "Score"])

df["Engine"] = pd.to_numeric( df["Engine"].str.replace("L", ""))
df = pd.get_dummies(df, columns=["Type", "Transmission", "Fuel"], dtype="int", drop_first=True)

y = df[["Price(£)"]]
x = df.drop("Price(£)", axis=1)

lm = LinearRegression()
model = lm.fit(x,y)
without_traintestsplit = model.score(x,y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

model2 = lm.fit(x_train, y_train)

main_model = model2.score(x_test, y_test)
