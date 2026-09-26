# Apply LogisticRegression Model
lr = LogisticRegression()
lr.fit(xre_train, yre_train)

lr_pred = lr.predict(xre_test)
lr_test = lr.score(xre_test, yre_test)

print("Accuracy:", accuracy_score(yre_test, lr_pred))
print(classification_report(yre_test, lr_pred, labels=[0, 1]))



# Top 10 Feature importances
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Get the coefficients from the Logistic Regression model
coefficients = lr.coef_[0]

# Create a pandas Series with coefficients and feature names
feature_coefficients = pd.Series(coefficients, index=xre_train.columns)

# Sort the features by the absolute value of their coefficients
sorted_coefficients = feature_coefficients.reindex(feature_coefficients.abs().sort_values(ascending=False).index)

# Select top N features (e.g., top 10) based on absolute coefficient values
top_n = 10
top_features_lr = sorted_coefficients.head(top_n)

# Create a bar plot of top feature coefficients with different colors for positive and negative coefficients
colors = ['red' if coef < 0 else 'blue' for coef in top_features_lr]
plt.figure(figsize=(10, 6))
top_features_lr.plot(kind='bar', color=colors)
plt.title(f'Top {top_n} Feature Importances (Coefficients) - Logistic Regression')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Confusion Matrix of Logistic Regression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming yre_test and lr_pred (from Logistic Regression) are available
cm_lr = confusion_matrix(yre_test, lr_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()



# ROC Curve of Logistic Regression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Get predicted probabilities
lr_probs = lr.predict_proba(xre_test)

# Calculate ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(2):  # Assuming binary classification (0 and 1)
    fpr[i], tpr[i], _ = roc_curve(np.array(yre_test) == i, lr_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))

for i in range(2):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class %d, area = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Logistic Regression (Class-wise)')
plt.legend(loc="lower right")
plt.show()



#Apply RandomForest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(xre_train, yre_train)

y_pred = rf.predict(xre_test)
accuracy_rf = accuracy_score(yre_test, y_pred)

print("Accuracy:", accuracy_rf)
print(classification_report(yre_test, y_pred, labels=[0, 1]))


# Feature importance of RandomForest
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances from the Random Forest model
feature_importances = pd.Series(rf.feature_importances_, index=xre_train.columns)

# Sort feature importances and select top N features
top_n = 10
top_features = feature_importances.sort_values(ascending=False).head(top_n)

# Create a bar plot with example color
plt.figure(figsize=(10, 6))
top_features.plot(kind='bar', color='#440154')

plt.title(f'Top {top_n} Feature Importances (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Confusion Matrix of RandomForest
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming yre_test and y_pred (from Random Forest) are available
cm = confusion_matrix(yre_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()


# ROC curve of RandomForest
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Get predicted probabilities
rf_probs = rf.predict_proba(xre_test)

# Calculate ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(2):  # Assuming binary classification (0 and 1)
    fpr[i], tpr[i], _ = roc_curve(np.array(yre_test) == i, rf_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))

for i in range(2):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class %d, area = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest (Class-wise)')
plt.legend(loc="lower right")
plt.show()




# Apply Decision Tree Model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(xre_train, yre_train)

y_pred = dt.predict(xre_test)
accuracy_dt = accuracy_score(yre_test, y_pred)

print("Accuracy:", accuracy_dt)
print(classification_report(yre_test, y_pred, labels=[0, 1]))


# Top 10 Feature importance of Decision Tree
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances from the Decision Tree model
feature_importances_dt = pd.Series(dt.feature_importances_, index=xre_train.columns)

# Sort feature importances and select top N features (e.g., top 10)
top_n = 10
top_features_dt = feature_importances_dt.sort_values(ascending=False).head(top_n)

# Create a bar plot of top feature importances
plt.figure(figsize=(10, 6))
top_features_dt.plot(kind='bar')
plt.title(f'Top {top_n} Feature Importances (Decision Tree)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Confusion Matrix of Decision Tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming yre_test and y_pred (from Decision Tree) are available
cm_dt = confusion_matrix(yre_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree')
plt.show()



#ROC Curve of Decision Tree
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Get predicted probabilities for the positive class (class 1)
dt_probs = dt.predict_proba(xre_test)

# Calculate ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(2):  # Assuming binary classification (0 and 1)
    fpr[i], tpr[i], _ = roc_curve(np.array(yre_test) == i, dt_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(8, 6))

for i in range(2):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class %d, area = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Decision Tree (Class-wise)')
plt.legend(loc="lower right")
plt.show()


# Apply of K-Nearest Neighbor
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(xre_train, yre_train)

y_pred_knn = knn_model.predict(xre_test)
accuracy_knn = accuracy_score(yre_test, y_pred_knn)

print("Accuracy:", accuracy_knn)
print(classification_report(yre_test, y_pred_knn, labels=[0, 1]))


# Confusion Matrix of KNN
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming yre_test and y_pred_knn (from KNN) are available
cm_knn = confusion_matrix(yre_test, y_pred_knn)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - KNN')
plt.show()



# ROC Curve of CNN
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Get predicted probabilities for the positive class (class 1)
knn_probs = knn_model.predict_proba(xre_test)

# Calculate ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(2):  # Assuming binary classification (0 and 1)
    fpr[i], tpr[i], _ = roc_curve(np.array(yre_test) == i, knn_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(8, 6))

for i in range(2):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class %d, area = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - KNN (Class-wise)')
plt.legend(loc="lower right")
plt.show()


# Apply of XGBoost Model
import xgboost as xgb

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(xre_train, yre_train)

y_pred_xgb = xgb_model.predict(xre_test)
accuracy_xgb = accuracy_score(yre_test, y_pred_xgb)

print("XGBoost Accuracy:", accuracy_xgb)
print("XGBoost Classification Report:")
print(classification_report(yre_test, y_pred_xgb, labels=[0, 1]))


# Top 10 feature importance of XGBoost
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances from the XGBoost model
feature_importances_xgb = pd.Series(xgb_model.feature_importances_, index=xre_train.columns)

# Sort feature importances and select top N features (e.g., top 10)
top_n = 10
top_features_xgb = feature_importances_xgb.sort_values(ascending=False).head(top_n)

# Create a bar plot of top feature importances
plt.figure(figsize=(10, 6))
top_features_xgb.plot(kind='bar')
plt.title(f'Top {top_n} Feature Importances (XGBoost)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Confusion Matrix of XGBoost
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming yre_test and y_pred_xgb (from XGBoost) are available
cm_xgb = confusion_matrix(yre_test, y_pred_xgb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.show()


#ROC Curve of XGBoost
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Get predicted probabilities for the positive class (class 1)
xgb_probs = xgb_model.predict_proba(xre_test)

# Calculate ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(2):  # Assuming binary classification (0 and 1)
    fpr[i], tpr[i], _ = roc_curve(np.array(yre_test) == i, xgb_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(8, 6))

for i in range(2):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class %d, area = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost (Class-wise)')
plt.legend(loc="lower right")
plt.show()



# Apply MLP Model
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(random_state=42, max_iter=300) # Increased max_iter for convergence
mlp_model.fit(xre_train, yre_train)

y_pred_mlp = mlp_model.predict(xre_test)
accuracy_mlp = accuracy_score(yre_test, y_pred_mlp)

print("MLP Accuracy:", accuracy_mlp)
print("MLP Classification Report:")
print(classification_report(yre_test, y_pred_mlp, labels=[0, 1]))


#Confusion Matrix of MLP
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming yre_test and y_pred_mlp (from MLP) are available
cm_mlp = confusion_matrix(yre_test, y_pred_mlp)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - MLP')
plt.show()


# ROC Curve of MLP
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Get predicted probabilities for the positive class (class 1)
mlp_probs = mlp_model.predict_proba(xre_test)

# Calculate ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(2):  # Assuming binary classification (0 and 1)
    fpr[i], tpr[i], _ = roc_curve(np.array(yre_test) == i, mlp_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(8, 6))

for i in range(2):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class %d, area = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - MLP (Class-wise)')
plt.legend(loc="lower right")
plt.show()






















