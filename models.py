import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from preprocess import separate_target_variable, load_w6, alternated_merge_for_lstm, load_w5, replace_missing_values_w6, \
    remove_constant_features, group_pre_frailty, min_max_scaling, add_fried_w6, preprocess_frailty_db, \
    joint_feature_selection_df
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import inspect

# Dictionary of classifiers to be used for the baseline
classifiers = {"SVM_linear": SVC(kernel='linear', C=0.2),
               "SVM_rbf": SVC(kernel='rbf', C=1.0),
               "MLP": MLPClassifier(hidden_layer_sizes=(100, 100, 75, 50, 50, 25,), alpha=0.001, max_iter=1000),
               "DT": DecisionTreeClassifier(max_depth=20),
               "RF": RandomForestClassifier(max_depth=20, n_estimators=20),
               "LR": LogisticRegression(max_iter=1000)
               }


def load_data(file_name, folder_path='data/best_features/', target_variable='FFP', index=None):
    """
    Loads a Dataframe and returns X and y

    :param index:
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


def get_metrics(classifier, X_train, X_test, y_train, y_test, epochs=20):
    """
    Computes accuracy, precision and F1 of a classifier

    :param classifier: {sklearn model} classifier to fit train data and compute metrics on test data
    :param X_train: {pandas DataFrame} (n_train_samples, n_features)
    :param X_test: {pandas DataFrame} (n_test_samples, n_features)
    :param y_train: {numpy array} (n_train_samples)
    :param y_test: {numpy array} (n_test_samples)
    :return: accuracy {float}, precision {float} and F1 score {float} of the classifier
    """
    if 'epochs' in inspect.getfullargspec(classifier.fit).args:
        classifier.fit(X_train, y_train, epochs=epochs)
    else:
        classifier.fit(X_train, y_train)
    accuracy = accuracy_score(y_true=y_test, y_pred=classifier.predict(X_test))
    precision = precision_score(y_true=y_test, y_pred=classifier.predict(X_test), average='macro')
    f1 = f1_score(y_true=y_test, y_pred=classifier.predict(X_test), average='macro')

    return accuracy, precision, f1


def get_classifiers_metrics(X_train, X_test, y_train, y_test, random_state=None, epochs=20):
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
            get_metrics(classifier=classifiers[clf], X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        epochs=epochs))
        # print(str(clf) + ":\n" + classification_report(y_test, classifiers[clf].predict(X_test)))
    metrics_df = pd.DataFrame.from_dict(data=metrics, orient='index', columns=['accuracy', 'precision', 'f1'])
    return metrics_df


def get_cv_metrics(X, y, scoring, cv=5, epochs=10, callbacks=None, verbose=0, random_state=None):
    metrics = {}
    for clf in classifiers:
        classifiers[clf].set_params(random_state=random_state)
        if 'epochs' in inspect.getfullargspec(classifiers[clf].fit).args:
            scores = cross_validate(classifiers[clf], X=X, y=y, scoring=scoring, cv=cv,
                                    fit_params={'epochs': epochs, 'callbacks': callbacks}, verbose=verbose)
        else:
            scores = cross_validate(classifiers[clf], X=X, y=y, scoring=scoring, cv=cv,
                                    fit_params=None, verbose=verbose)
        metrics[clf] = [scores['test_' + str(metric)].mean() for metric in scoring]
        # print(str(clf) + ":\n" + classification_report(y_test, classifiers[clf].predict(X_test)))
    metrics_df = pd.DataFrame.from_dict(data=metrics, orient='index', columns=scoring)
    return metrics_df


def get_lstm_metrics(frailty_df, y, older_df, epochs=20, batch_size=64, lstm_units=50, best_features_selection=True,
                     k=30, random_state=None, verbosity=0, cv=5):
    X, y = alternated_merge_for_lstm(older_df=older_df, y=y, frailty_df=frailty_df,
                                     best_features_selection=best_features_selection, k=k)
    print(X.shape)
    X = np.array(X).reshape(X.shape[0] // 2, 2, X.shape[1])
    print("final after reshape: " + str(X.shape))
    # labels = y
    labels = to_categorical(y, num_classes=2)
    print("labels after to_categorical: " + str(labels.shape))
    acc_per_fold = []
    loss_per_fold = []
    f_score_per_fold = []

    model = Sequential()
    model.add(LSTM(units=lstm_units * 2, activation='relu', return_sequences=True, input_shape=(2, X.shape[2])))
    model.add(LSTM(units=lstm_units, activation='relu', return_sequences=True))
    model.add(LSTM(units=lstm_units, activation='relu', return_sequences=True))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    kfold = KFold(n_splits=cv, shuffle=True)
    for train, test in kfold.split(X, labels):
        history = model.fit(X[train], labels[train],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbosity)
        # X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
        # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        # scores = cross_validate(model, X=X, y=labels, scoring=scoring, cv=cv,
        #                         fit_params={'epochs': epochs}, verbose=verbose)

        scores = model.evaluate(X[test], labels[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        print("labels[test]: " + str(labels[test].shape))
        print("model.predict(X[test]): " + str(model.predict(X[test]).shape))
        print("model.predict(X[test][0]): " + str(model.predict(X[test][0])))

        # rounded_labels = np.argmax(labels[test], axis=1)
        y_pred = model.predict(X[test])  # , batch_size=batch_size, verbose=1)
        y_pred_bool = np.argmax(y_pred, axis=1)
        for i in range(y_pred.shape[0]):
            if y_pred_bool[i] == 0:
                y_pred[i] = [1, 0]
            else:
                y_pred[i] = [0, 1]

        print("y_pred: " + str(y_pred.shape))

        # metrics_df = pd.DataFrame.from_dict(classification_report(labels[test], y_pred, output_dict=True), orient='index')
        f_score_per_fold.append(float(classification_report(labels[test], y_pred, output_dict=True)['macro avg']['f1-score']))

    metrics_df = pd.DataFrame.from_dict(
        {'accuracy': np.mean(acc_per_fold), 'loss': np.mean(loss_per_fold), 'f1-macro':np.mean(f_score_per_fold)},
        orient='index')
    return X, metrics_df


if __name__ == '__main__':
    print("Starting process...")
    # Create log folder and callbacks

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # WAVE 6 FRAILTY DETECTION
    # ---------------------------------------------
    # Load data
    # df = load_w6(core_data_path="data/raw/wave_6_elsa_data_v2.tab",
    #              nurse_data_path="data/raw/wave_6_elsa_nurse_data_v2.tab", remove_duplicates=True, index_col="idauniq")
    # # Calculate Fried Frailty Phenotype
    # X, y = add_fried_w6(elsa_w6_merged=df, drop_columns=True, drop_rows=True)
    # # Preprocess the data
    # X, y = preprocess_frailty_db(X=X, y=y, replace_missing_value=True, regex_list=None, replace_negatives=True,
    #                              replace_nan=True, rm_constant_features=True, min_max=True, group_frailty=True)
    # # Select the best features
    # k = 30
    # selection_functions = ["chi2", "f_classif", "mutual_info_classif"]  # ["chi2", "f_classif", "mutual_info_classif"]
    # X = joint_feature_selection_df(X=X, y=y, score_functions=selection_functions, k=k)

    # OR directly load preprocessed and best-features-selected data
    data_file = 'w6_frailty_selected_50_features_mutual_info_classif.tab'
    folder_path = 'data/best_features/'
    X, y = load_data(file_name=data_file, folder_path=folder_path, target_variable="FFP", index="idauniq")
    # Set Random state and number of epochs
    random_state = 10
    epochs = 30
    cv = 10
    scoring = ['accuracy', 'precision_macro', 'f1_macro']
    # Train models
    metrics_df = get_cv_metrics(X=X, y=y, scoring=scoring, cv=cv, epochs=epochs, callbacks=None,
                                verbose=20)
    # Save results (only works if data are loaded from pre-precessed file)
    metrics_df.to_csv(
        "data/metrics/wave_6/metrics_of_" + data_file.split('.', 1)[0] + '_random_state_' + str(random_state) + '.tab',
        sep='\t', quoting=3, escapechar='\\')
    print('WAVE 6 FINISHED')
    #
    # # WAVE 5 FRAILTY PREDICTION
    # # ---------------------------------------------
    # # Load wave 6 data
    # df_w6, y = load_data(file_name="wave_6_frailty_FFP_data.tab", folder_path="data/raw/", target_variable="FFP",
    #                      index="idauniq")
    #
    # # Load wave 5 data
    # data_file = "wave_5_elsa_data_v4.tab"
    # X = load_w5(core_data_path="data/raw/" + str(data_file), index_col='idauniq', acceptable_features=None,
    #             acceptable_idauniq=df_w6.index)
    # print("DATA LOADED")
    # # Filter frailty variable and sort the data
    # y = y.loc[X.index]
    # y.sort_index(inplace=True)
    # X.sort_index(inplace=True)
    # # Preprocess the data
    # X, y = preprocess_frailty_db(X=X, y=y, replace_missing_value=True, regex_list=None, replace_negatives=True,
    #                              replace_nan=True, rm_constant_features=True, min_max=True, group_frailty=True)
    # print("DATA PREPROCESSED")
    # # Select the best features
    # k = 20
    # selection_functions = ["chi2", "f_classif", "mutual_info_classif"]  # ["chi2", "f_classif", "mutual_info_classif"]
    # X = joint_feature_selection_df(X=X, y=y, score_functions=selection_functions, k=k, print_selected=True)
    # print("FEATURES SELECTED")
    # # Set Random state and number of epochs
    # random_state = 30
    # epochs = 100
    # cv = 10
    # # Train models
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # metrics_df = get_cv_metrics(X=X, y=y, scoring=['accuracy', 'precision_macro', 'f1_macro'],
    #                             random_state=random_state, epochs=epochs, cv=cv)
    # # Save results
    # metrics_df.to_csv(
    #     "data/metrics/wave_5/metrics_of_" + data_file.split('.', 1)[0] + '_random_state_' + str(random_state) + '.tab',
    #     sep='\t', quoting=3, escapechar='\\')
    # print('WAVE 5 FINISHED')

    # # WAVE 5 AND 6 LSTM FRAILTY DETECTION
    # # ---------------------------------------------
    # # Load wave 6 data
    # df = load_w6(core_data_path="data/raw/wave_6_elsa_data_v2.tab",
    #              nurse_data_path="data/raw/wave_6_elsa_nurse_data_v2.tab", remove_duplicates=True, index_col="idauniq")
    # # Calculate Fried Frailty Phenotype
    # X, y = add_fried_w6(elsa_w6_merged=df, drop_columns=True, drop_rows=True)
    # # Preprocess the data
    # X, y = preprocess_frailty_db(X=X, y=y, replace_missing_value=True, regex_list=None, replace_negatives=True,
    #                              replace_nan=True, rm_constant_features=True, min_max=True, group_frailty=True)
    # # Load wave 5 data
    # df_w5 = load_w5(core_data_path="data/raw/wave_5_elsa_data_v4.tab", acceptable_features=np.array(X.columns),
    #                 acceptable_idauniq=np.array(X.index))
    # # Set bet features, random state and number of epochs
    # k = 25
    # random_state = 20
    # epochs = 30
    # batch_size = 50
    # cv = 100
    # # Train LSTM
    # X, lstm_metrics_df = get_lstm_metrics(frailty_df=X, y=y, older_df=df_w5, epochs=epochs, batch_size=batch_size,
    #                                       best_features_selection=True, k=k, cv=cv)
    # # Save results
    # lstm_metrics_df.to_csv("data/metrics/lstm/metrics_of_lstm_" + str(X.shape[2]) + "_features.tab", quoting=3,
    #                        escapechar='\\', sep='\t')
    print("All done!")
