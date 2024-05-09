import numpy as np
import pandas as pd
from skrebate import multisurf

pd.options.mode.chained_assignment = None


def load_w6(core_data_path="data/raw/wave_6_elsa_data_v2.tab",
            nurse_data_path="data/raw/wave_6_elsa_nurse_data_v2.tab", index_col="idauniq", remove_duplicates=True):
    """
    Loads wave 6 core data and nurse visit data from .tab files, and merges them into a single pandas DataFrame.

    :param core_data_path: {string} path to Elsa wave 6 core raw data
    :param nurse_data_path: {string} path to Elsa wave 6 nurse visit raw data
    :param index_col: {string} name of the index column that both entry dataframes should have. Default='idauniq'
    :param remove_duplicates: {Boolean} if True, remove duplicate columns from the two dataframes. If False, the duplicate columns from the second df will have the suffix '_y'. Default=True
    :return: merged {DataFrame} (n_samples, n_features), merged dataframe on the index column
    """
    nurse = pd.read_csv(nurse_data_path, sep='\t', lineterminator='\n', header=(0), index_col=index_col,
                        low_memory=False)
    main = pd.read_csv(core_data_path, sep='\t', lineterminator='\n', header=(0), index_col=index_col, low_memory=False)
    merged = main.merge(right=nurse, on=index_col, suffixes=('', '_y'))
    if remove_duplicates:
        merged.drop(merged.filter(regex='_y$').columns, axis=1, inplace=True)
    return merged


def load_w5(core_data_path="data/raw/wave_5_elsa_data_v4.tab", index_col='idauniq', acceptable_features=None,
            acceptable_idauniq=None, drop_frailty_columns=None):
    """
    Loads wave 5 data from a .tab file and returns them in a pandas DataFrame.

    :param core_data_path: {string} path to Elsa wave 5 core raw data
    :param index_col: {string} name of the index column that should coincide with the index of wave 6. Default='idauniq'
    :param acceptable_features: {array} list of acceptable features to load. Alternative to drop_frailty_columns. Default=None
    :param acceptable_idauniq: {array} list of acceptable indexes, useful to only load patients who were still present in wave 6 and got the frailty level compudet. Default=None
    :param drop_frailty_columns: {array} list of variable to drop from the DataFrame. Default=None
    :return: df {DataFrame} (n_samples, n_features), wave 5 dataframe
    """
    df = pd.read_csv(core_data_path, sep='\t', lineterminator='\n', header=(0), index_col=index_col, low_memory=False)
    if drop_frailty_columns is not None:
        df = df.drop(labels=drop_frailty_columns, axis=1)
    if (acceptable_features is None) and (acceptable_idauniq is None):
        return df
    if not (acceptable_features is None):
        acceptable_features = np.array(set(df.columns) & set(acceptable_features))
        df = df.loc[:, acceptable_features]
        if acceptable_idauniq is None:
            return df
    df = df.filter(items=acceptable_idauniq, axis=0)
    return df


def frailty_level_w6(sex, height, weight, grip_strength, walking_time, exhaustion, activity_level):
    """
    Calculates Fried's Frailty Phenotype (FFP) starting from wave 6 ELSA database, obtained by merging core data and nurse visit data.

    :param sex: {int} 1 = Male, 2 = Female (indsex from core data)
    :param height: {float} measure in cm (height from core data)
    :param weight: {float} measure in kg (weight from nurse visit)
    :param grip_strength: {float} measure with dominant hand in kg (max between mmgsd1, mmgsd2 and mmgsd3 from nurse visit)
    :param walking_time: {float} measure in m/s (min between MMWlkA and MMWlkB from core data)
    :param activity_level: {int} from 3 (max activity) to 12 (sum of HeActa HeActb and HeActc from core data)
    :param exhaustion:{int}  from 2 (exhausted) to 4 (sum of PScedB and PScedH from core data)
    :return: {int} -1 if unable to calculate; 0 if not frail; 1 if pre-frail; 2 if frail.
    """
    met_criteria = 0

    # Check for invalid ELSA data
    if (height <= 0) | (weight <= 0) | (grip_strength <= 0) | (walking_time <= 0.05) | (activity_level < 3) | (
            exhaustion < 2):
        return -1

    # Weight loss
    bmi = weight / pow(height / 100, 2)
    if bmi < 18.5:
        met_criteria += 1

    # Grip strength and walking speed
    walking_speed = 2.44 / walking_time
    if sex == 1:
        if (grip_strength <= 29) | ((bmi > 24) & (grip_strength <= 30)) | ((bmi > 26) & (grip_strength <= 31)) | (
                (bmi > 28) & (grip_strength <= 32)):
            met_criteria += 1
        if (walking_speed <= 0.65) | ((height > 173) & (walking_speed <= 0.76)):
            met_criteria += 1
    else:
        if (grip_strength <= 17) | ((bmi > 23.1) & (grip_strength <= 17.3)) | ((bmi > 26.1) & (grip_strength <= 18)) | (
                (bmi > 29) & (grip_strength <= 21)):
            met_criteria += 1
        if (walking_speed <= 0.65) | ((height > 159) & (walking_speed <= 0.76)):
            met_criteria += 1

    # Activity level
    if activity_level >= 10:
        met_criteria += 1

    # Exhaustion
    if exhaustion <= 3:
        met_criteria += 1

    if met_criteria == 0:
        return 0
    elif met_criteria <= 2:
        return 1
    else:
        return 2


def add_fried_w6(elsa_w6_merged, drop_columns=False, drop_rows=False):
    """
    Adds Fried's Frailty Phenotype ('FFP') column to wave 6 ELSA merged database.

    :param elsa_w6_merged: {pandas DataFrame} (n_samples, n_features) ELSA wave 6 with core data and nurse visit merged
    :param drop_columns: {Bool} if True deletes the columns used for FFP computation, default=False
    :param drop_rows: {Bool} if True deletes the rows for which FFP could not be calculated, default=False
    :return: elsa_w6_merged {pandas DataFrame} (n_samples, n_features + 1) original dataframe with FFP column added
    """

    elsa_w6_merged['FFP'] = elsa_w6_merged.apply(lambda row: frailty_level_w6(sex=row.indsex,
                                                                              height=row.height,
                                                                              weight=row.weight,
                                                                              grip_strength=max(row.mmgsd1, row.mmgsd2,
                                                                                                row.mmgsd3),
                                                                              walking_time=min(row.MMWlkA, row.MMWlkB),
                                                                              exhaustion=row.PScedB + row.PScedH,
                                                                              activity_level=row.HeActa + row.HeActb + row.HeActc
                                                                              ), axis=1)

    if drop_columns:
        elsa_w6_merged = elsa_w6_merged.drop(columns=['height', 'weight', 'HTOK', 'HTVAL', 'WTOK', 'WTVAL',
                                                      'BMI', 'BMIOK', 'BMIVAL', 'BMIOBE',
                                                      'mmgsd1', 'mmgsd2', 'mmgsd3', 'mmgsn1', 'mmgsn2', 'mmgsn3',
                                                      'MMWlkA', 'MMWlkB',
                                                      'PScedB', 'PScedH',
                                                      'HeActa', 'HeActb', 'HeActc'])

    if drop_rows:
        elsa_w6_merged = elsa_w6_merged.loc[elsa_w6_merged['FFP'] >= 0]

    y = elsa_w6_merged.loc[:, 'FFP']
    elsa_w6_merged.drop('FFP', axis=1, inplace=True)
    return elsa_w6_merged, y


def group_pre_frailty(y):
    """
    Given an array of FFP (target) variable, groups frail and pre-frail subjects under the same value. Assumes there are no negative values.

    :param y: {array} (n_samples) frailty variable (0=non-frail, 1=pre-frail, 2=frail)
    :return: {array} (n_samples) with binary target variable (0=non-frail, 1=pre-frail or frail) instead of multiclass
    """

    y = y.apply(lambda x: min(x, 1))
    return y


def load_data(file_name, folder_path='data/best_features/', target_variable='FFP', index=None):
    """
    Loads a Dataframe and returns X and y

    :param file_name: {str} name of the .tab file containing the dataframe with target variable
    :param folder_path: {str} path to the folder containing the file
    :param target_variable: {str} name of the target variable y inside the df in file
    :param index: {str} name of the index column
    :return: X {pandas DataFrame} (n_samples, n_features), y {numpy array} (n_samples)
    """
    if index is None:
        df = pd.read_csv(filepath_or_buffer=folder_path + file_name, sep='\t', lineterminator='\n', header=0,
                         low_memory=False)
    else:
        df = pd.read_csv(filepath_or_buffer=folder_path + file_name, sep='\t', lineterminator='\n', header=0,
                         low_memory=False, index_col=index)
    return separate_target_variable(df=df, target_variable=target_variable)


def replace_missing_values_w6(frailty_dataframe, regex_list=None, replace_negatives=None, replace_nan=None):
    """
    Replaces nan entries, str entries and negative floats with 0

    :param frailty_dataframe: {pandas DataFrame} (n_samples, n_features) X
    :param regex_list: {list of str} list of regular expressions to be replaced by nan, only matters if replace_missing_values=True, default=None
    :param replace_negatives: {float, int, nan} value with which to replace negatives values of X, default=None
    :param replace_nan: {float, int} value with which to replace nan values of X, default=None
    :return: {pandas DataFrame} (n_samples, n_features) with replaced values
    """

    if regex_list is None:
        regex_list = ["\s",
                      ".*:.*:.*", ".*-.*-.*",
                      "[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz]",
                      ]
    frailty_dataframe.replace(to_replace=regex_list, value=np.nan, inplace=True, regex=True)
    frailty_dataframe = frailty_dataframe.astype(float)
    if replace_negatives is not None:
        frailty_dataframe.replace(to_replace=[i for i in range(-1, -10, -1)] + [str(i) for i in range(-1, -10, -1)],
                                  value=replace_negatives, inplace=True)
    if replace_nan is not None:
        frailty_dataframe.replace(to_replace=np.nan, value=replace_nan, inplace=True)
    return frailty_dataframe


def remove_constant_features(X):
    """
    Removes constant features from the dataframe

    :param X: {pandas DataFrame} (n_samples, n_features)
    :return: {pandas DataFrame} (n_samples, n_features_not_removed)
    """

    return X.loc[:, X.apply(pd.Series.nunique) != 1]


def min_max_scaling(X):
    """
    Scales values of the dataframe between 0 and 1 with min-max logic, except for the target frailty variable

    :param X: {pandas DataFrame} (n_samples, n_features) df to scale
    :return: {pandas DataFrame} (n_samples, n_features) scaled following min-max logic
    """

    X_scaled = (X - X.min()) / (X.max() - X.min())
    X_scaled.fillna(0, inplace=True)
    return X_scaled


def preprocess_frailty_db(X, y, replace_missing_value=True, regex_list=None, replace_negatives=None, replace_nan=None,
                          rm_constant_features=False, min_max=True, group_frailty=True):
    """

    :param X: {pandas DataFrame} (n_samples, n_features) df to preprocess
    :param y: {np array} (n_samples) target labels
    :param replace_missing_value: {Bool} if True replaces missing values of X, default=True
    :param regex_list: {list of str} list of regular expressions to be replaced by nan, only matters if replace_missing_values=True, default=None
    :param replace_negatives: {float, int, nan} value with which to replace negatives values of X, default=None
    :param replace_nan: {float, int} value with which to replace nan values of X, default=None
    :param rm_constant_features: {Bool} if True removes constant features from X, default=False
    :param min_max: {Bool} if True scales X using min-max logic, default=True
    :param group_frailty: {Bool} if True groups 1 and 2 values of y, default=True
    :return: {pandas DataFrame} (n_samples, n_features), {np array} (n_samples): processed df, target labels
    """
    if replace_missing_value:
        X = replace_missing_values_w6(frailty_dataframe=X, regex_list=regex_list, replace_negatives=replace_negatives,
                                      replace_nan=replace_nan)
    if rm_constant_features:
        X = remove_constant_features(X)
    if min_max:
        X = min_max_scaling(X=X)
    if group_frailty:
        y = group_pre_frailty(y=y)
    return X, y


def separate_target_variable(df, target_variable="FFP"):
    """
    Separates target variable column from the rest of the df.

    :param df: {pandas DataFrame} (n_samples, n_features + 1) df containing target variable
    :param target_variable: {string} name of the target variable column within df, default='FFP'
    :return: {pandas DataFrame} (n_samples, n_features), {np array} (n_samples): X, y
    """

    y = df.loc[:, target_variable]
    df.drop(target_variable, axis=1, inplace=True)
    return df, y


def multisurf_feature_selection(X, y, n_features=50, discrete_threshold=20, n_jobs=1, save_features=False, file_path=None):
    """

    :param X: {pandas DataFrame} (n_samples, n_features)
    :param y: {np array} (n_samples) target labels
    :param n_features: {int} number of features to select, default=50
    :param discrete_threshold: {int} max number of values to consider a feature categorical, default=20
    :param n_jobs: {int} number of parallel jobs, default=1
    :param save_features: {Bool} if True, saves list of selected features in 'file_path', default=False
    :param file_path: {str} path to save the features list, only works if save_features=True, default=None
    :return: {pandas DataFrame} (n_samples, n_features_selected), {np array} (n_samples): selected_X, y
    """
    selector = multisurf.MultiSURF(n_features_to_select=n_features, discrete_threshold=discrete_threshold, n_jobs=n_jobs)
    X_test = X.copy().to_numpy()
    y_test = y.copy().to_numpy()
    selector = selector.fit(X=X_test, y=y_test)
    selected_vars = list(selector.top_features_)
    X_surf = X.iloc[:, selected_vars[:n_features]]
    if save_features:
        vars = pd.DataFrame(data=np.array(X_surf.columns).reshape(1, n_features))
        vars.to_csv(file_path, sep='\t', escapechar='\\')
    return X_surf, y


if __name__ == '__main__':
    # Load wave 6 and 5
    X_w6, y_w6 = load_data(file_name="wave_6_frailty_FFP_data.tab", folder_path="data/raw/", target_variable="FFP",
                     index="idauniq")
    data_file_w5 = "wave_5_elsa_data_v4.tab"
    X_w5 = load_w5(core_data_path="data/raw/" + str(data_file_w5), index_col='idauniq', acceptable_features=None,
                acceptable_idauniq=X_w6.index, drop_frailty_columns=False)
    y_w5 = y_w6.loc[X_w5.index]
    y_w5.sort_index(inplace=True)
    X_w5.sort_index(inplace=True)
    # Preprocess data
    X_w6, y_w6 = preprocess_frailty_db(X=X_w6, y=y_w6, replace_missing_value=True, regex_list=None,
                                       replace_negatives=np.nan, replace_nan=None, rm_constant_features=True, min_max=True, group_frailty=True)
    X_w5, y_w5 = preprocess_frailty_db(X=X_w5, y=y_w5, replace_missing_value=True, regex_list=None,
                                       replace_negatives=np.nan, replace_nan=None, rm_constant_features=True,
                                       min_max=True, group_frailty=True)
    # Select and save best variables
    n_features=50
    multisurf_feature_selection(X=X_w6, y=y_w6, n_features=n_features, discrete_threshold=20, n_jobs=-1, save_features=True,
                                file_path="data/best_features/wave_6_features.tab")
    multisurf_feature_selection(X=X_w5, y=y_w5, n_features=n_features, discrete_threshold=20, n_jobs=-1, save_features=True,
                                file_path="data/best_features/wave_5_features.tab")
