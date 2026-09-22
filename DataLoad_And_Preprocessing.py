#This code for Mount
from google.colab import drive
drive.mount('/content/drive')

#Installing and importing various types of packages and libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,roc_curve, auc)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#Data Load
df = pd.read_csv("/content/drive/MyDrive/ML/diabetes_Binary_health_indicators_BRFSS2015.csv")


# Check and Remove duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")
df.drop_duplicates(inplace=True)
print(f"Number of rows after removing duplicates: {df.shape[0]}")

# Check for outliers using IQR for all features
outlier_counts = {}
for feature in df.columns:
    if df[feature].dtype in ['int64', 'float64']: # Check only numerical columns
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_counts[feature] = len(outliers)
    else:
        outlier_counts[feature] = "Not numerical"

print("Number of outliers per feature (using IQR for numerical features):")
for feature, count in outlier_counts.items():
    print(f"{feature}: {count}")

# Handle outliers in identified features by capping using IQR
features_to_cap = ['Veggies', 'PhysHlth', 'CholCheck', 'BMI', 'Stroke', 'HeartDiseaseorAttack', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'DiffWalk'] # Combine features from both cells

for feature in features_to_cap:
    if df[feature].dtype in ['int64', 'float64']:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Capping
        df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)

print("Outliers in identified features have been capped.")

#Checking imbalanced data
import matplotlib.pyplot as plt
# Get the counts of each class in the 'Diabetes_binary' column
diabetes_counts = df['Diabetes_binary'].value_counts()

# Create a doughnut chart
plt.figure(figsize=(8, 8))
plt.pie(diabetes_counts, labels=diabetes_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)

# Draw a circle at the center to create the doughnut effect
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Distribution of Diabetes Outcome (0: Non-Diabetic, 1: Diabetic)')
plt.show()

# Converting data Imbalance to Balance
from imblearn.over_sampling import SMOTE
# Separate features (X) and target (y)
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Class distribution after SMOTE:")
print(y_resampled.value_counts())

# visualize the distribution
plt.figure(figsize=(8, 5))
y_resampled.value_counts().plot(kind='bar')
plt.title('Distribution of Disease')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

#Use StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_resampled)
scaled_features = scaler.transform(X_resampled)
X_resampled = pd.DataFrame(scaled_features,columns=X_resampled.columns)
X_resampled.head(10)

#Train Test split
xre_train, xre_test, yre_train, yre_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
print("Shape of X_train:", xre_train.shape)
print("Shape of X_test:", xre_test.shape)
print("Shape of y_train:", yre_train.shape)
print("Shape of y_test:", yre_test.shape)