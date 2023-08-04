# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# %%
data = pd.read_csv('heart.csv')
data.head()

# %%
data.isnull().sum()

# %%
y = data['condition']
X = data.drop(['condition'], axis = 1)

# %%
X.shape

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=10)

# %%
model = RandomForestClassifier()

# %%
model.fit(X_train, y_train)

# %%
model.oob_score

# %%
model.score(X_train, y_train)


# %%
model.score(X_test, y_test)

# %%
y_pred = model.predict(X_test)
y_pred

# %%
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")



