import pandas as pd
import numpy as np
from preprocess import separate_target_variable, preprocess_frailty_db, load_w5, load_data
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import inspect
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Dictionary of classifiers to be used for the baseline
classifiers = {"SVM_linear": [SVC(kernel='linear'), {'C': [0.1, 1, 10]}],
               "SVM_rbf": [SVC(kernel='rbf'), {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1]}],
               # "MLP": [MLPClassifier(),
               #         {'hidden_layer_sizes': [(100, 50,), (100, 75, 25,), (100, 100, 75, 50, 25,)],
               #          'activation': ['tanh', 'relu'], 'alpha': [0.001, 0.0001], 'max_iter': [2000]}],
               # "DT": [DecisionTreeClassifier(), {'max_depth': [5, 10, 20]}],
               # "RF": [RandomForestClassifier(), {'max_depth': [5, 10, 20], 'n_estimators': [20, 50, 100]}],
               "LR": [LogisticRegression(), {'C': [0.1, 1, 10], 'max_iter': [2000]}]
               }


def get_cv_metrics(X, y, scoring=['accuracy', 'precision_macro', 'f1_macro', 'recall_macro'], voting_classifier=False,
                   cv=10, epochs=100, verbose=0, random_state=None, results_file_path=None, model_file_path=None,
                   n_jobs=1):
    """
    Trains and saves the models specified in the dictionary 'classifiers'. Saves and returns the result metrics.
    :param X: {pandas DataFrame} (n_samples, n_features)
    :param y: {np array} (n_samples) target labels
    :param scoring: {list of strings} scoring values. Recommended default=['accuracy', 'precision_macro', 'f1_macro', 'recall_macro']
    :param voting_classifier: {Bool} if True also trains an ensembler voting classifier from the previous models, default=False
    :param cv: {int} number of folds in cross validation, default=10
    :param epochs: {int} number of epochs in training, default=100
    :param verbose: {int} level of verbosity of the fitting function, default=0
    :param random_state: {int} seed, default=None
    :param results_file_path: {str} path where to save result metrics, default=None
    :param model_file_path: {str} path where to save models, default=None
    :param n_jobs: {int} number of parallel jobs
    :return: {pandas DataFrame} final_results, result metrics
    """
    results = {}
    for clf in classifiers:
        classifiers[clf][0].set_params(random_state=random_state)
        crossv = KFold(n_splits=cv, shuffle=True)
        grid_search = GridSearchCV(estimator=classifiers[clf][0], param_grid=classifiers[clf][1], scoring=scoring,
                                   cv=crossv, verbose=verbose, refit='f1_macro', n_jobs=n_jobs)
        if 'epochs' in inspect.getfullargspec(classifiers[clf][0].fit).args:
            grid_search.fit(X=X, y=y, epochs=epochs)
        else:
            grid_search.fit(X=X, y=y)

        # Set the best parameters to the models
        best_params = grid_search.best_params_
        classifiers[clf][0].set_params(**best_params)

        # Results
        metrics_df = pd.DataFrame.from_dict(data=grid_search.cv_results_)
        results[clf] = metrics_df.loc[
            metrics_df["rank_test_f1_macro"] == 1, ["params", "mean_test_accuracy", "mean_test_precision_macro",
                                                    "mean_test_f1_macro", "mean_test_recall_macro"]].values.tolist()[0]

        # NEW: save best model
        if model_file_path is not None:
            filename = str(model_file_path) + "_" + str(clf) + '.sav'
            pickle.dump(grid_search.best_estimator_, open(filename, 'wb'))

    # Voting classifier
    if voting_classifier:
        vc_estimators = [(name, clf) for name, (clf, _) in classifiers.items()]
        clf = VotingClassifier(estimators=vc_estimators, voting='hard', n_jobs=n_jobs)
        y_pred = cross_val_predict(clf, X, y, cv=cv)

        # Calculate voting classifier metrics
        mean_precision = precision_score(y, y_pred)
        mean_accuracy = accuracy_score(y, y_pred)
        mean_f1_score = f1_score(y, y_pred)
        mean_recall = recall_score(y, y_pred)

        # Update results
        results["VC"] = [np.nan, mean_accuracy, mean_precision, mean_f1_score, mean_recall]

    final_results = pd.DataFrame.from_dict(data=results, orient='index',
                                           columns=["params", "accuracy", "precision_macro", "f1_macro",
                                                    "recall_macro"])
    # Save Results
    if results_file_path is not None:
        final_results.to_csv(results_file_path + '.tab', sep='\t', escapechar='\\', mode='a')

    return final_results


if __name__ == '__main__':
    print("Starting process...")

    # WAVE 6 FRAILTY PREDICTION
    # ---------------------------------------------
    # Load wave 6 data
    X, y = load_data(file_name="wave_6_frailty_FFP_data.tab", folder_path="data/raw/", target_variable="FFP",
                     index="idauniq")

    # Load selected variables
    prediction_multisurf_variables = pd.read_csv("data/best_features/wave_6_features.tab",
                                                 sep='\t', escapechar='\\')['0'].tolist()
    X = X.loc[:, prediction_multisurf_variables]

    # Preprocess the data
    X, y = preprocess_frailty_db(X=X, y=y, replace_missing_value=True, regex_list=None, replace_negatives=np.nan,
                                 replace_nan=None, rm_constant_features=True, min_max=True, group_frailty=True)

    # NEW: eliminate specific features
    X.drop(list(X.filter(regex='ff.*')), axis=1, inplace=True)

    # Select the best_grid_search parameters
    scoring = ['accuracy', 'precision_macro', 'f1_macro', 'recall_macro']
    folds = 10
    seed = 10
    epochs = 1000

    # Train models and save results
    saved_model_path = "data/models/detection/test"
    saved_results_path = "data/metrics/detection_test_results"
    get_cv_metrics(X=X, y=y, scoring=scoring, voting_classifier=True, random_state=seed, epochs=epochs, cv=folds,
                   results_file_path=saved_results_path, model_file_path=saved_model_path, n_jobs=-1)

    print('WAVE 6 FINISHED')

    # -----------------------------------------------------------------------------------------------------------------
    # Comment all section above this line to get only prediction results.
    # -----------------------------------------------------------------------------------------------------------------

    # WAVE 5 FRAILTY PREDICTION
    # ---------------------------------------------
    # Load wave 6 data
    df_w6, y = load_data(file_name="wave_6_frailty_FFP_data.tab", folder_path="data/raw/", target_variable="FFP",
                         index="idauniq")

    # Load wave 5 data
    data_file = "wave_5_elsa_data_v4.tab"
    X = load_w5(core_data_path="data/raw/" + str(data_file), index_col='idauniq', acceptable_features=None,
                acceptable_idauniq=df_w6.index, drop_frailty_columns=None)
    prediction_multisurf_variables = pd.read_csv("data/best_features/wave_5_features.tab",
                                                 sep='\t', escapechar='\\')['0'].tolist()
    X = X.loc[:, prediction_multisurf_variables]

    # Filter frailty variable and sort the data
    y = y.loc[X.index]
    y.sort_index(inplace=True)
    X.sort_index(inplace=True)

    # Preprocess the data
    X, y = preprocess_frailty_db(X=X, y=y, replace_missing_value=True, regex_list=None, replace_negatives=np.nan,
                                 replace_nan=None, rm_constant_features=True, min_max=True, group_frailty=True)

    # NEW: eliminate specific features
    X.drop(list(X.filter(regex='ff.*')), axis=1, inplace=True)

    # Select the best_grid_search features
    scoring = ['accuracy', 'precision_macro', 'f1_macro', 'recall_macro']
    folds = 10
    seed = 10
    epochs = 1000

    # Train models and save results
    saved_model_path = "data/models/prediction/test"
    saved_results_path = "data/metrics/prediction_test_results"
    get_cv_metrics(X=X, y=y, scoring=scoring, voting_classifier=True, random_state=seed, epochs=epochs, cv=folds,
                   results_file_path=saved_results_path, model_file_path=saved_model_path, n_jobs=-1)

    print('WAVE 5 FINISHED')
