from sklearn.datasets import load_iris
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as skRF
from classification.models import RandomForest

iris = load_iris(as_frame=True)
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

model = RandomForest(n_estimators=1, random_state=42, max_depth=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("This:", matthews_corrcoef(y_test, y_pred))

sk_model = skRF(n_estimators=1, random_state=42, max_depth=1)
sk_model.fit(X_train, y_train)
y_pred = sk_model.predict(X_test)
print("sklearn:", matthews_corrcoef(y_test, y_pred))
