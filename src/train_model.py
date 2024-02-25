import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import config

df = pd.read_csv(config.SOURCE)
df.to_csv(config.CURRENT, index=False)

X = df.drop([config.TARGET], axis=1)
y = df[config.TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Logistic Regression \n")
print("accuracy: ", accuracy_score(y_test, y_pred))

pickle.dump(model, open("model/model.pkl", "wb"))
