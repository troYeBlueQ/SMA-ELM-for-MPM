from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score

def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc_value = auc(*roc_curve(y_test, proba)[:2])
    acc = accuracy_score(y_test, model.predict(X_test))
    return proba, auc_value, acc

def train_rf(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc_value = auc(*roc_curve(y_test, proba)[:2])
    acc = accuracy_score(y_test, model.predict(X_test))
    return proba, auc_value, acc
