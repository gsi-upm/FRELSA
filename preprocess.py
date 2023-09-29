import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression


def load_w6(core_data_path="wave_6_elsa_data_v2.tab", nurse_data_path="wave_6_elsa_nurse_data_v2.tab"):
    """
    Loads core data and nurse visit data from wave 6 and merges them into a single pandas dataframe.
    """

    nurse = pd.read_csv(nurse_data_path, sep='\t', lineterminator='\n', header=(0))
    main = pd.read_csv(core_data_path, sep='\t', lineterminator='\n', header=(0), low_memory=False)
    merged = main.merge(right=nurse, on='idauniq')
    return merged


def frailty_level_w6(sex, height, weight, grip_strength, walking_time, exhaustion, activity_level):
    """
    Calculates Fried's frailty phenotype starting from wave 6 ELSA merged database.
    Returns: -1 if unable to calculate;
    1 if subject is not frail;
    2 if subject is pre-frail;
    3 if subject is frail.

    sex: 1 = Male, 2 = Female (indsex)
    height: measure in cm (height)
    weight: measure in kg (weight)
    grip_strength: max measurment with dominant hand in kg (max between mmgsd1, mmgsd2 and mmgsd3)
    walking_time: measure in m/s (min between MMWlkA and MMWlkB)
    activity level: from 3 (max activity) to 12 (sum of HeActa HeActb and HeActc)
    exhaustion: from 2 (exhausted) to 4 (sum of PScedB and PScedH)
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
        return 1
    elif met_criteria <= 2:
        return 2
    else:
        return 3


def add_fried_w6(elsa_w6_merged, drop_columns=False, drop_rows=False):
    """
    Calculates Fried's Frailty Phenotype starting from wave 6 ELSA merged database.
    Returns the dataframe with an added column called 'FFP', with following values:
    -1 if unable to calculate;
    1 if subject is not frail;
    2 if subject is pre-frail;
    3 if subject is frail;

    elsa_w6_merged: DataFrame of ELSA wave 6 with core data and nurse visit merged
    drop_columns: if True deletes the columns used for FFP computation
    drop_rows: if True deletes the rows for which FFP could not be calculated
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
        elsa_w6_merged.drop(columns=['height', 'weight',
                                     'mmgsd1', 'mmgsd2', 'mmgsd3',
                                     'MMWlkA', 'MMWlkB',
                                     'PScedB', 'PScedH',
                                     'HeActa', 'HeActb', 'HeActc'])

    if drop_rows:
        elsa_w6_merged = elsa_w6_merged.loc[elsa_w6_merged['FFP'] > 0]

    return elsa_w6_merged


def feature_selection_w6(frailty_dataframe, frailty_variable='FFP', score_func=chi2, k=100):
    """
    Returns a dataframe with k best feature and the frailty variable
    :param frailty_dataframe: df from which features are selected
    :param frailty_variable: frailty target variable contained in frailty_dataframe
    :param score_func: what function to use to find the k-best variables
    :param k: number of variables to select
    :return: dataframe
    """
    variables = list(frailty_dataframe.columns.values)
    variables.remove(frailty_variable)
    x = frailty_dataframe.loc[:, variables]
    y = frailty_dataframe.loc[:, frailty_variable]
    select = SelectKBest(score_func=score_func, k=k)
    best_features_dataframe = select.fit_transform(x, y)
    print(type(best_features_dataframe), best_features_dataframe.shape, best_features_dataframe[-1])
    return best_features_dataframe


if __name__ == '__main__':
    print("Starting process...")

    core_data_path = "data/wave_6_elsa_data_v2.tab"
    nurse_data_path = "data/wave_6_elsa_nurse_data_v2.tab"
    fried = add_fried_w6(elsa_w6_merged=load_w6(core_data_path=core_data_path, nurse_data_path=nurse_data_path), drop_columns=True, drop_rows=True)
    fried.to_csv('data/wave_6_frailty_FFP_data.tab', sep='\t', index=False, quoting=3, escapechar='\\')

    # fried = pd.read_csv(filepath_or_buffer='data/wave_6_frailty_FFP_data.tab', sep='\t', lineterminator='\n', header=(0), low_memory=False)
    # feature_selection_w6(frailty_dataframe=fried)

    print("All done!")
