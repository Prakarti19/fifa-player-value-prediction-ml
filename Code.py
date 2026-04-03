#%%
# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler

#%%

# 2. LOAD DATASET
file_path = "MachineLearning_EDA.csv"
df = pd.read_csv(file_path)

# 3. BASIC DATA UNDERSTANDING
print(" FIRST 5 ROWS")
display(df.head())

print("\n SHAPE OF THE DATASET (rows, columns)")
print(df.shape)

print("\n COLUMN NAMES")
print(df.columns.tolist())

print("\n DATA TYPES")
print(df.dtypes)

print("\n DATAFRAME INFO")
print(df.info())

print("\n SUMMARY STATISTICS (NUMERICAL)")
display(df.describe().T)

# Categorical summary
categorical_cols = df.select_dtypes(include=["object"]).columns
print("\n CATEGORICAL FEATURES SUMMARY")
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df[col].value_counts().head(10))
    print("Missing values:", df[col].isna().sum())

#%%

# 3. FINDING MISSING VALUES

print("\n MISSING VALUES IN EACH COLUMN")
missing_values = df.isnull().sum()
display(missing_values[missing_values > 0])

numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns



# 5. DUPLICATE CHECK
print("\n DUPLICATE ROWS:")
duplicates = df.duplicated().sum()
print("Total duplicate rows:", duplicates)

#%%

# 4. EXPLORATORY DATA ANALYSIS (EDA)
print("MISSING VALUES")
missing = df.isnull().sum()
missing = missing[missing > 0]
display(missing)

if not missing.empty:
    missing.plot(kind="bar", color="salmon")
    plt.ylabel("Count of missing values")
    plt.title("Missing Values by Column")
    plt.show()

# Target distribution
target = "value_eur"

plt.figure(figsize=(10,5))
sns.histplot(df[target], kde=True, bins=40)
plt.title("Distribution of Player Market Value (value_eur)")
plt.xlabel("value_eur")
plt.show()

print("Skewness:", df[target].skew())

plt.figure(figsize=(10,5))
sns.histplot(np.log1p(df[target]), kde=True, bins=40)
plt.title("Distribution After log1p(value_eur)")
plt.xlabel("log(value_eur)")
plt.show()



# Outlier check
plt.figure(figsize=(12,5))
sns.boxplot(x=df[target])
plt.title("Boxplot for value_eur (Outlier Check)")
plt.show()
 
#%%

# 5. FEATURE ENGINEERING
CURRENT_YEAR = 2025

# Remaining Contract Years
df['contract_end_year'] = pd.to_numeric(df['contract_valid_until'], errors='coerce')
df['remaining_contract_years'] = df['contract_end_year'] - CURRENT_YEAR
df = df.drop(columns=['contract_end_year'])

# Years at Club
df['joined_year'] = df['joined'].str.extract(r'(\d{4})').astype(float)
df['years_at_club'] = CURRENT_YEAR - df['joined_year']
df = df.drop(columns=['joined_year'])

# BMI
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# Potential Growth
df['potential_growth'] = df['potential'] - df['overall']

# Main Position
df['main_position'] = df['player_positions'].str.split(',').str[0].str.strip()
# Position group (manually grouped roles)
def categorize_position(pos):
    if pos == 'GK':
        return 'Goalkeeper'
    elif pos in ['CB', 'LB', 'RB', 'LWB', 'RWB', 'LCB', 'RCB']:
        return 'Defender'
    elif pos in ['CDM', 'CM', 'CAM', 'LM', 'RM', 'LCM', 'RCM', 'LDM', 'RDM']:
        return 'Midfielder'
    elif pos in ['ST', 'CF', 'LW', 'RW', 'LF', 'RF', 'LS', 'RS']:
        return 'Attacker'
    else:
        return 'Other'

df['position_group'] = df['main_position'].apply(categorize_position)

#%%

# 5. MISSING VALUE HANDLING 
# standardize missing-value tokens
df.replace(['', ' ', 'NA', 'N/A', 'na', '?', 'None', 'none'], np.nan, inplace=True)

# drop columns that are 100% NaN (if any)
all_na_cols = df.columns[df.isna().mean() == 1.0].tolist()
if all_na_cols:
    print("Dropping fully empty columns:", all_na_cols)
    df.drop(columns=all_na_cols, inplace=True)

# Numerical → median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if df[col].isna().any():
        med = df[col].median()
        # if med is NaN (all values NaN), skip (we already dropped full-NaN columns)
        if not np.isnan(med):
            df[col].fillna(med, inplace=True)
        else:
            # as fallback fill with 0
            df[col].fillna(0, inplace=True)

# Categorical → mode (safe: check if mode exists)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    if df[col].isna().any():
        modes = df[col].mode(dropna=True)
        if len(modes) > 0:
            df[col].fillna(modes[0], inplace=True)
        else:
            df[col].fillna('Unknown', inplace=True)

# Final check
print("Total missing values remaining:", df.isnull().sum().sum())

# Correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()
corr_target = corr["value_eur"].sort_values(ascending=False)

print("\nCORRELATION OF FEATURES WITH TARGET (value_eur)")
display(corr_target)

# CROSS‑CORRELATION (NUMERICAL FEATURES)
# Select numerical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Compute correlation matrix
cross_corr = df[num_cols].corr()

print("CROSS‑CORRELATION MATRIX")
display(cross_corr)

# Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(cross_corr, cmap="coolwarm", annot=False)
plt.title("Cross‑Correlation Heatmap (Numerical Features)", fontsize=14)
plt.show()


plt.figure(figsize=(14,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

print("TOP CORRELATED FEATURES WITH TARGET")
display(corr_target.head(15))

# Scatterplots for top features
top_features = corr_target.drop(target).head(6).index

for col in top_features:
    plt.figure(figsize=(8,4))
    sns.scatterplot(x=df[col], y=df[target], alpha=0.4)
    sns.regplot(x=df[col], y=df[target], scatter=False, color="red")
    plt.title(f"{col} vs value_eur")
    plt.show()

# Categorical inspection
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("CATEGORICAL COLUMNS")
print(cat_cols)

example_cats = [c for c in ["position_group", "nationality_name", "club_name", "preferred_foot"]
                if c in df.columns]

for col in example_cats:
    print(f"\nTop categories for {col}:")
    display(df[col].value_counts().head(10))

    top10 = df[col].value_counts().index[:10]
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df[col][df[col].isin(top10)], y=df[target][df[col].isin(top10)])
    plt.xticks(rotation=45)
    plt.title(f"value_eur Distribution by {col} (Top 10)")
    plt.show()
    
#%%

# 6. FEATURE SELECTION (Correlation + PCA Combined)

target = "value_eur"

# SAFE CORRELATION CALCULATION (No MemoryError)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Compute only the correlations of each feature with target (1 column at a time)
corr_target = df[numeric_cols].corrwith(df[target]).abs()


# CORRELATION‑BASED FILTERING
corr_threshold = 0.40
corr_selected = corr_target[corr_target > corr_threshold].index.tolist()

# Remove target if included
corr_selected = [col for col in corr_selected if col != target]

print("\n Features selected based on correlation (> 0.40):")
print(corr_selected)
print("Count:", len(corr_selected))

# PCA‑BASED FILTERING
pca_data = df[corr_selected].copy()

# Additional safety: remove zero‑variance features
pca_data = pca_data.loc[:, pca_data.var() > 0]

# Perform PCA keeping 95% variance
pca = PCA(n_components=0.95, random_state=42)
pca.fit(pca_data)

# Feature contributions = sum of squared loadings
pca_importance = np.sum(pca.components_**2, axis=0)

# Keep features with strong PCA contributions
pca_threshold = 0.02   # TUNEABLE
pca_selected = [
    feat for feat, score in zip(pca_data.columns, pca_importance)
    if score > pca_threshold
]

print("\n Features selected by PCA (importance > 0.02):")
print(pca_selected)
print("Count:", len(pca_selected))

# UNION OF CORRELATION + PCA
final_selected_features = sorted(list(set(corr_selected).union(set(pca_selected))))

print("\n FINAL SELECTED FEATURES AFTER CORR ∪ PCA:")
print(final_selected_features)
print("Total:", len(final_selected_features))

# FINAL CLEANED DATASET
final_df = df[final_selected_features + [target]]

print("\nFinal dataset shape:", final_df.shape)

features_to_drop = corr_target[corr_target.abs() < 0.4].index.tolist()
features_to_drop = [f for f in features_to_drop if f != target]

df.drop(columns=features_to_drop, inplace=True)

print("\n Dropped features (|corr_target| < 0.5):")
print(features_to_drop)

#%%
# 7. ENCODING CATEGORICAL VARIABLES
cat_cols = final_df.select_dtypes(include=["object"]).columns.tolist()

# Label encoding for limited categories
from sklearn.preprocessing import LabelEncoder
label_enc_cols = ["preferred_foot", "position_group", "main_position"]

le = LabelEncoder()
for col in label_enc_cols:
    if col in df.columns:
        final_df[col] = le.fit_transform(df[col])
        print(f"Label encoded: {col}")

# One-hot encoding for the remaining categorical columns
one_hot_cols = [col for col in cat_cols if col not in label_enc_cols]

final_df = pd.get_dummies(final_df, columns=one_hot_cols, drop_first=True)
print("One-hot encoded columns:", one_hot_cols)
print("Final shape after encoding:", df.shape)

#%%

# 8. SCALING NUMERIC FEATURES 
from sklearn.preprocessing import StandardScaler

num_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'value_eur']  # exclude target

scaler = StandardScaler()
final_df[num_cols] = scaler.fit_transform(final_df[num_cols])

print("Scaled numeric columns:", len(num_cols))

#%%

# 9. TRAIN–TEST SPLIT (80/20)
X = final_df.drop(columns=["value_eur"])
y = final_df["value_eur"]

# Split the dataset: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTrain/Test split completed.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 10. MODEL TRAINING AND EVALUATION (WITH LATENCY)
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import time

# Adjusted R² function
def adjusted_r2(r2, X):
    n = X.shape[0]
    k = X.shape[1]
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

# Dictionary to store results
results = {
    "Model": [],
    "MAE": [],
    "RMSE": [],
    "R2": [],
    "Adjusted_R2": [],
    "Latency (sec)": []   # ← NEW COLUMN
}

# List of models
models = [
    ("Linear Regression", LinearRegression()),
    ("Lasso Regression", Lasso(alpha=0.01)),
    ("Decision Tree", DecisionTreeRegressor(random_state=42)),
    ("Random Forest", RandomForestRegressor(random_state=42)),
    ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
    ("Support Vector Regressor", SVR())
]

# Train + evaluate each model
for name, model in models:
    # Train ONLY on training data
    model.fit(X_train, y_train)

    # Measure latency for prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    latency = time.time() - start_time

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2(r2, X_test)

    # Store results
    results["Model"].append(name)
    results["MAE"].append(mae)
    results["RMSE"].append(rmse)
    results["R2"].append(r2)
    results["Adjusted_R2"].append(adj_r2)
    results["Latency (sec)"].append(latency)

# Final comparison table
results_df = pd.DataFrame(results)


print("MODEL PERFORMANCE COMPARISON")
print(results_df)

#%%

# 10. VISUAL MODEL COMPARISON (BAR CHARTS)
import matplotlib.pyplot as plt

# Make plots look cleaner
plt.style.use("seaborn-v0_8")

metrics_to_plot = ["MAE", "RMSE", "R2", "Adjusted_R2", "Latency (sec)"]

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["Model"], results_df[metric])
    plt.title(f"Comparison of {metric} Across Models")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#%%

# 11. SINGLE SUMMARY PLOT (ALL MODELS IN ONE FIGURE)

metrics = ["MAE", "RMSE", "R2", "Adjusted_R2"]

models = results_df["Model"].values
x = np.arange(len(models))

scaled = MinMaxScaler().fit_transform(results_df[metrics])
scaled_df = pd.DataFrame(scaled, columns=metrics)

plt.figure(figsize=(14,6))
for i, metric in enumerate(metrics):
    plt.bar(x + i*0.2, scaled_df[metric], width=0.2, label=metric)

plt.xticks(x + 0.3, models, rotation=30)
plt.title("Normalized Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()
#%%










