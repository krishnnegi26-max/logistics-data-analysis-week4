# Week 4: Predictive Modeling and Optimization in Logistics Systems
# Simulated educational dataset; not real operational data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
n = 500
distance = np.random.randint(10, 1000, n)
weight = np.random.randint(1, 500, n)
mode = np.random.choice(["Road", "Rail", "Air"], n)
weather = np.random.choice(["Clear", "Rain", "Storm"], n)
traffic = np.random.randint(1, 6, n)
speed = np.select([mode == "Air", mode == "Rail", mode == "Road"], [550, 70, 48])
weather_delay = np.select([weather == "Clear", weather == "Rain", weather == "Storm"], [0, 2.5, 7])
handling = np.select([mode == "Air", mode == "Rail", mode == "Road"], [2, 4.5, 1.5])
noise = np.random.normal(0, 2.8, n)
delivery = np.maximum(1, distance/speed + handling + traffic*0.7 + weather_delay + weight/500*1.8 + noise)
df = pd.DataFrame({
    "Shipment_ID": range(1, n+1), "Distance_km": distance,
    "Shipment_Weight_kg": weight, "Transport_Mode": mode,
    "Weather_Condition": weather, "Traffic_Level": traffic,
    "Delivery_Time_Hours": np.round(delivery, 2)
})

print("Shape:", df.shape)
print("Missing values:\\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df.describe())

# Visualizations
sns.histplot(data=df, x="Delivery_Time_Hours", bins=20, kde=True)
plt.title("Distribution of Delivery Time"); plt.tight_layout(); plt.show()
sns.boxplot(data=df, x="Transport_Mode", y="Delivery_Time_Hours")
plt.title("Delivery Time by Transport Mode"); plt.tight_layout(); plt.show()
sns.scatterplot(data=df, x="Distance_km", y="Delivery_Time_Hours", hue="Transport_Mode")
plt.title("Distance vs Delivery Time"); plt.tight_layout(); plt.show()

X = df[["Distance_km", "Shipment_Weight_kg", "Transport_Mode", "Weather_Condition", "Traffic_Level"]]
y = df["Delivery_Time_Hours"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

numeric = ["Distance_km", "Shipment_Weight_kg", "Traffic_Level"]
categorical = ["Transport_Mode", "Weather_Condition"]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=250, max_depth=10,
                                           min_samples_leaf=2, random_state=42)
}
results = {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {"MAE": mean_absolute_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                     "R2": r2_score(y_test, pred), "pipeline": pipe}
    print(name, "MAE:", round(results[name]["MAE"], 3),
          "RMSE:", round(results[name]["RMSE"], 3),
          "R2:", round(results[name]["R2"], 3))

cv = KFold(n_splits=5, shuffle=True, random_state=42)
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    scores = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(name, "CV MAE scores:", np.round(scores, 3),
          "Mean:", round(scores.mean(), 3), "Std:", round(scores.std(), 3))

best = results["Random Forest"]["pipeline"]
pred = best.predict(X_test)
plt.scatter(y_test, pred, alpha=0.7)
lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
plt.plot([lo, hi], [lo, hi], "--")
plt.xlabel("Actual Delivery Time (Hours)"); plt.ylabel("Predicted Delivery Time (Hours)")
plt.title("Actual vs Predicted Delivery Time"); plt.tight_layout(); plt.show()

rf = best.named_steps["model"]
names = best.named_steps["preprocessor"].get_feature_names_out()
imp = pd.DataFrame({"Feature": names, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
print(imp.to_string(index=False))
sns.barplot(data=imp, x="Importance", y="Feature")
plt.title("Feature Importance"); plt.tight_layout(); plt.show()

sample = pd.DataFrame({"Distance_km":[600], "Shipment_Weight_kg":[250],
                       "Transport_Mode":["Road"], "Weather_Condition":["Storm"],
                       "Traffic_Level":[4]})
print("Example prediction:", round(best.predict(sample)[0], 2), "hours")
