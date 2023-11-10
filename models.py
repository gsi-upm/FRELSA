import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from preprocess import separate_target_variable
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier


# Dictionary of classifiers to be used for the baseline
classifiers = {"SVM_linear": SVC(kernel='linear', C=0.2),
               "SVM_rbf": SVC(kernel='rbf', C=1.0),
               "MLP": MLPClassifier(alpha=0.001, max_iter=1000),
               "DT": DecisionTreeClassifier(max_depth=5),
               "RF": RandomForestClassifier(max_depth=5, n_estimators=20),
               "LR": LogisticRegression(max_iter=1000)
               }


def load_data(file_name, folder_path='data/best_features/', target_variable='FFP', index=None):
    """
    Loads a Dataframe and returns X and y

    :param file_name: {string} name of the .tab file containing the dataframe with target variable
    :param folder_path: {string} path to the folder containing the file
    :param target_variable: {string} name of the target variable y inside the df in file
    :return: X {pandas DataFrame} (n_samples, n_features), y {numpy array} (n_samples)
    """
    if index is None:
        df = pd.read_csv(filepath_or_buffer=folder_path + file_name, sep='\t', lineterminator='\n', header=0,
                         low_memory=False)
    else:
        df = pd.read_csv(filepath_or_buffer=folder_path + file_name, sep='\t', lineterminator='\n', header=0,
                         low_memory=False, index_col=index)
    return separate_target_variable(df=df, target_variable=target_variable)


def get_metrics(classifier, X_train, X_test, y_train, y_test):
    """
    Computes accuracy, precision and F1 of a classifier

    :param classifier: {sklearn model} classifier to fit train data and compute metrics on test data
    :param X_train: {pandas DataFrame} (n_train_samples, n_features)
    :param X_test: {pandas DataFrame} (n_test_samples, n_features)
    :param y_train: {numpy array} (n_train_samples)
    :param y_test: {numpy array} (n_test_samples)
    :return: accuracy {float}, precision {float} and F1 score {float} of the classifier
    """
    classifier.fit(X_train, y_train)
    accuracy = accuracy_score(y_true=y_test, y_pred=classifier.predict(X_test))
    precision = precision_score(y_true=y_test, y_pred=classifier.predict(X_test))
    f1 = f1_score(y_true=y_test, y_pred=classifier.predict(X_test))
    return accuracy, precision, f1


def get_classifiers_metrics(X_train, X_test, y_train, y_test, random_state=None):
    """
    Compute metrics of all the classifiers in the 'classifiers' dictionary, and returns a df with classifiers as 'index' and metrics as 'columns'.

    :param X_train: {pandas DataFrame} (n_train_samples, n_features)
    :param X_test: {pandas DataFrame} (n_test_samples, n_features)
    :param y_train: {numpy array} (n_train_samples)
    :param y_test: {numpy array} (n_test_samples)
    :param random_state: {None or int} pass an int for reproducible results across multiple function calls.
    :return: {pandas DataFrame} (n_classifiers, 3=n_metrics) table with classifiers as rows and metrics as columns.
    """
    metrics = {}
    for clf in classifiers:
        classifiers[clf].set_params(random_state=random_state)
        metrics[clf] = list(
            get_metrics(classifier=classifiers[clf], X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test))
    metrics_df = pd.DataFrame.from_dict(data=metrics, orient='index', columns=['accuracy', 'precision', 'f1'])
    return metrics_df


if __name__ == '__main__':
    print("Starting process...")
    # data_file = 'w6_frailty_selected_56_features_chi2+f_classif+mutual_info_classif.tab'
    data_file = 'w6_frailty_selected_50_features_mutual_info_classif.tab'
    folder_path = 'data/best_features/'
    frailty_variable = "FFP"
    random_state = 10
    X, y = load_data(file_name=data_file, folder_path=folder_path, target_variable=frailty_variable, index="idauniq")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    metrics_df = get_classifiers_metrics(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, random_state=random_state)
    metrics_df.to_csv("data/metrics/metrics_of_" + data_file.split('.', 1)[0] + '_random_state_' + str(random_state) + '.tab')

    print("All done!")