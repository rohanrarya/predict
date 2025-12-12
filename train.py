from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import joblib


data = pd.read_csv('data/data.csv')

# ---- REMOVE OUTLIERS (BEST WAY TO REDUCE MAE) ----
q1 = data["price"].quantile(0.01)
q99 = data["price"].quantile(0.80)
data = data[(data["price"] > q1) & (data["price"] < q99)]

x = data.drop(columns=["price"])
y = data["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
def evaluate_model(size,age,room,distance):
    df = pd.DataFrame([[size,age,room,distance]], columns=["size","age","rooms","distance"])
    df_scaled = scaler.transform(df)
    
    return model.predict(df_scaled)[0]



joblib.dump(evaluate_model, 'models/evaluate.joblib')



