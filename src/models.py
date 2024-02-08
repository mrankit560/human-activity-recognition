from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import pandas as pd
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow

# Function to load dataset from Google Drive
def load_dataset_from_gdrive(gdrive_url, test_size=0.2, random_state=12):
    gdrive_id = gdrive_url.split('/')[-2]
    destination_path = 'mhealth_raw_data.csv'
    gdown.download(f'https://drive.google.com/uc?id={gdrive_id}', destination_path, quiet=False)
    data = pd.read_csv(destination_path)
    # Assuming your dataset needs some preprocessing; otherwise, adjust as necessary.
    data = data[data['Activity'] != 0]
    data_req = data.groupby(by=['Activity']).sample(n=5000, random_state=random_state)
    Y = data_req['Activity']
    X = data_req.drop(['Activity', 'subject'], axis=1)
    X = pd.DataFrame(StandardScaler().fit_transform(X))
    mlflow.log_param("dataset_path", destination_path)
    mlflow.log_param("dataset_shape", data_req.shape)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

# Google Drive link
gdrive_url = 'https://drive.google.com/file/d/14RkZYl9BdzFaOpZimL9FPRpIrWGEsbMY/view'

# Your model training functions, updated to use load_dataset_from_gdrive
# Replace logisticRegression, decision_tree, random_forest, etc., with the updated functions.

def logisticRegression():
    mlflow.sklearn.autolog()
    experiment_id = mlflow.set_experiment("Logistic Regression Model")
    with mlflow.start_run(run_name='logistic regression') as run:
        X_train, X_test, y_train, y_test = load_dataset_from_gdrive(gdrive_url)
        lr = LogisticRegression(penalty="l2", solver='liblinear')
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        print(y_pred)
        metrics = mlflow.sklearn.eval_and_log_metrics(lr,
                                                      X_test,
                                                      y_test,
                                                      prefix="test_")
        mlflow.end_run()


def decision_tree():
    mlflow.sklearn.autolog()
    experiment_id = mlflow.set_experiment("Decision Trees")
    with mlflow.start_run(run_name='decision tree') as run:
        X_train, X_test, y_train, y_test = load_dataset_from_gdrive(gdrive_url)
        parameters = {'min_samples_split': range(2, 5, 10),
                      'max_depth': range(2, 5, 10)}
        clf_tree = DecisionTreeClassifier()
        clf = GridSearchCV(clf_tree, parameters)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = mlflow.sklearn.eval_and_log_metrics(clf,
                                                      X_test,
                                                      y_test,
                                                      prefix="test_")
        mlflow.end_run()


def decision_tree_with_KFold(n_splits=10):
    experiment_id = mlflow.set_experiment("Decision Trees with" +
                                          "K-Fold Cross Validation")
    with mlflow.start_run(run_name='decision tree') as run:
        X_train, X_test, y_train, y_test = load_dataset_from_gdrive(gdrive_url)
        clf_tree = DecisionTreeClassifier()
        k_fold = KFold(n_splits)
        mlflow.log_param("n_splits", n_splits)
        accuracy = cross_val_score(clf_tree, X_train, y_train, cv=k_fold)
        y_pred = cross_val_predict(clf_tree, X_test, y_test)
        mlflow.log_metric("training_score", accuracy.mean())
        accuracy = accuracy_score(y_pred, y_test)
        mlflow.log_metric("test_score", accuracy)
        mlflow.end_run()


def random_forest():
    experiment_id = mlflow.set_experiment("Random Forest")
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name='random forest') as run:
        X_train, X_test, y_train, y_test = load_dataset_from_gdrive(gdrive_url)
        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        metrics = mlflow.sklearn.eval_and_log_metrics(classifier,
                                                      X_test,
                                                      y_test,
                                                      prefix="test_")
        mlflow.end_run()


def xgboost_classification():
    experiment_id = mlflow.set_experiment("XGBoost")
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name='xgboost') as run:
        X_train, X_test, y_train, y_test = load_dataset_from_gdrive(gdrive_url)
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        metrics = mlflow.sklearn.eval_and_log_metrics(classifier,
                                                      X_test,
                                                      y_test,
                                                      prefix="test_")
        mlflow.end_run()


def neural_networks():
    experiment_id = mlflow.set_experiment("Neural Networks")
    mlflow.keras.autolog()
    with mlflow.start_run(run_name='neural networks') as run:
        X_train, X_test, y_train, y_test = load_dataset_from_gdrive(gdrive_url)
        model = Sequential()
        model.add(Dense(units=64,
                        kernel_initializer='normal',
                        activation='sigmoid',
                        input_dim=X_train.shape[1]))
        model.add(Dropout(0.2))
        model.add(Dense(units=13,
                  kernel_initializer='normal',
                  activation='softmax'))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        mo_fitt = model.fit(X_train, y_train, epochs=75,
                            validation_data=(X_test, y_test))
        mlflow.end_run()

if __name__ == '__main__':
    neural_networks()
