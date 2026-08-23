# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

# %%
df = pd.read_csv("heart_disease_uci.csv")

# %%
df.head()

# %%
missing_values = df.isnull().sum()
print(missing_values)

# %%
df = df.drop(['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'] , axis= 1)

# %%
df.head()

# %%
missing_values =df.isnull().sum()
print(missing_values)

# %%
df.shape

# %%
df_cleaned = df.dropna()

# %%
df_cleaned.shape

# %%
missing_values =df_cleaned.isnull().sum()
print(missing_values)

# %%
x = df_cleaned.drop('num', axis=1)
y = df_cleaned['num']

# %%
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size = 0.2 , random_state= 42)

# %%
x_train

# %%
x_test

# %%
clf_model = RandomForestClassifier()
clf_model.fit( x_train, y_train)

# %%
y_pred = clf_model.predict( x_test)

# %%
y_pred

# %%
y_test

# %%
cm = confusion_matrix( y_test, y_pred)


# %%
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.title("confusion Matrix")
plt.show()

# %%
print(classification_report(y_test , y_pred))

# %%


# %%


# %%


# %%


# %%



