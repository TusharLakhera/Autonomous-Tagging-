model = RidgeClassifier(tol=1e-2, solver="lsqr")
model.fit(X_train_1, y_train)
predicted = model.predict(X_test_1)
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted, target_names=target_names))
