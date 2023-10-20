import numpy as np
import pandas as pd
from CFSmethod import CFS
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFE
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

pd.options.mode.chained_assignment = None

function_dict = {
    "chi2": chi2,
    "f_classif": f_classif,
    "mutual_info_classif": mutual_info_classif,
}

variables_dict = {
    'hastro': "LUNG: Whether been admitted to hospital with a heart complaint in the past month?",
    'LungEx': "LUNG: Derived: Excluded from lung function test",
    'NoLung': "LUNG: Respondent not eligible for a lung function test",
    'NoAttLF1': "LUNG: Why unable to take reading: Respondent is breathless",
    'NoAttLF2': "LUNG: Why unable to take reading: Respondent is unwell",
    'NoAttLF3': "LUNG: Why unable to take reading: Respondent upset/anxious/nervous",
    'NoAttLF9': "LUNG: Why unable to take reading: Other reason",
    'LFSmHr': "LUNG: How many hours ago last smoked",
    'inhaler': "LUNG: Whether used an inhaler, puffer or any medication for breathing in the last 24 hours",
    'inhalhrs': "LUNG: How many hours ago last used an inhaler, puffer or any medication for breathing",
    'chestinf': "LUNG: Whether had any respiratory infections such as influenza, pneumonia, bronchitis or a severe "
                "cold, in the past three weeks",
    'htfvc': "LUNG: Highest technically satisfactory value for FVC",
    'PRFVC': "LUNG: Predicted value for FVC",
    'PCFVC': "LUNG: FVC as percentage of predicted FVC",
    'htfev': "LUNG: Highest technically satisfactory value for FEV",
    'PRFEV': "LUNG: Predicted value for FEV",
    'PCFEV': "LUNG: FEV as percentage of predicted FEV",
    'HTPEF': "LUNG: Highest technically satisfactory value for PEF",
    'PRPEF': "LUNG: Predicted value for PEF",
    'PCPEF': "LUNG: PEF as percentage of predicted PEF",
    'Quality': "LUNG: Derived: Outcome from lung function software (A - F)",
    'SYSVAL': "(D) Valid Mean Systolic BP",
    'DIAVAL': "(D) Valid Mean Diastolic BP",
    'PULVAL': "",
    'MAPVAL': "",
    'BSOUTC': "",
    'HTOK': "",
    'HTVAL': "",
    'WTOK': "",
    'WTVAL': "",
    'BMI': "",
    'BMIOK': "",
    'BMIVAL': "",
    'BMIOBE': "",
    'WSTOKB': "",
    'WSTVAL': "",
    'MMRROC': "",
    'MMFTRE2': "",
    'bpconst': "",
    'bswill': "",
    'w6nurwt': "",
    'w6bldwt': "",
    'vismon': "",
    'visyear': "",
    'finstat': "",
    'Indobyr_y': "",
    'indager\r': "",

    'painhh': "Whether the father of the respondent is in the household",
    'WhoSo1': "No-one other than respondent and interviewer in the room",
    'futype': "Financial unit type (1=single, 2=couple, but finances separate, 3=couple with joint finances)",
    'HeOpt96': "Eye: no diagnosis newly reported ",
    'hedia96': "CVD: no new diagnosis reported",
    'heaid96': "Aids used: not use any of listed aids (walking or special eating or personal alarm)",
    'HeBow': "Incontinence: whether problems controlling bowels in last 12 months",
    'CaHnDa': "Received help with at least one task from daughter (COMPUTED from CAHIN --> informal help)",
    'CaHnFr': "Received help with at least one task from friend (COMPUTED from CAHIN --> informal help)",
    'CaHnNe': "Received help with at least one task from neighbour (COMPUTED from CAHIN --> informal help)",
    'CaHnHM': "Received help with at least one task from a council handyman (COMPUTED from CAHFO --> formal help)",
    'CaHrs09': "COMPUTED : Received help from Grandchild B and asked about hours helped",
    'CaHrs12': "COMPUTED : Received help from Sister B and asked about hours helped",
    'CaHrs13': "COMPUTED : Received help from Sister C and asked about hours helped",
    'CaHrs19': "COMPUTED : Received help from Other relative C and asked about hours helped",
    'CaHrs21': "COMPUTED : Received help from Friend B and asked about hours helped",
    'CaHrs26': "COMPUTED : Received help from Home care worker/ home help/ personal assistant A and asked about hours helped",
    'CaHrs32': "COMPUTED : Received help from a cleaner and asked about hours helped",
    'CaHrs34': "COMPUTED : Received help from a member of staff at the care/nursing home and asked about hours helped",
    'wpdesc': "Whether answer to wpdes (Employment situation) was recoded post-interview from text answer",
    'WpPHI': "Covered by private health insurance (in own name or through a family member)",
    'WpExW': " Expect to get pension from scheme that former spouse/civil partner contrib. to?",
    'Iahdbc': "receiving any [of these health] or disability benefits at the moment?",
    'IaFuel': "Did you (or your spouse) receive a Winter Fuel Payment in the last year",
    'Iaregp': "Were any regular payments received from people living elsewhere (respondent)",
    'Iapar': "Were any regular payments received from people living elsewhere (partner)",
    'iapkm96': "Lump sums received (respondent/spouse) in last year: none of these (merged var)",
    'HoRet': "Is accommodation retirement housing?",
    'hofuemel': "Fuels used in household for heating or other purpose - electricity (merged var)",
    'HoVW': "Is this (1st) vehicle a car, a van or a motorbike?",
    'cfwhonon': "No other person present in room (for cognitive function test)",
    'PScedB': "!!!!!!!!!!!Whether felt everything they did during past week was an effort",
    'PScedG': "Whether felt sad much of the time during past week",
    'scptr1': "I read a daily newspaper",
    'scptr6': "I own a mobile phone",
    'scinp1': "Places respondent has used the internet or email in last 3 months: At home",
    'q33m09': "Whether sought help/advice regarding sex life from: Have not sought any help",
    'indsex': "Definitive sex variable: priority disex, dhsex",
    'cuffsize': "BPRESS: Cuff size used",
    'full2': "BPRESS: Set of BP readings are complete",
    'clotb': "BLOOD: Whether has clotting disorder",
    'fit': "BLOOD: Whether fitted in last 5 years",
    'samptak': "BLOOD: Any blood samples taken (incl DNA samples)",
    'DoneWst': "WAIST: DoneWst [waist measurement completed?]",
    'mmssre': "BALANCE: Outcome of side-by-side stand",
    'mmstsc': "BALANCE: Whether respondent feels it is safe to attempt semi-tandem stand",
    'hasurg': "LUNG: Whether had abdominal or chest surgery in the past 3 months",
    'LFTB': "LUNG: Whether currently taking any medications for the treatment of tuberculosis?",
    'LFSmok': "LUNG: Whether smoked in the last 24 hours",
    'LFWill': "Consent to Lung Function",
    'WPAskD': "Computed : Ask WpJdo or not",
    'WPAskE': "Computed : Ask WpEst or not",
    'hofuelel': "Fuels used in household for heating or other purpose - electricity",
    'difbpcno': "BPRESS: Problems taking BP readings: No problems taking blood pressure",



    'FFP': "Fried's Frailty Phenotype",
}


def load_w6(core_data_path="data/raw/wave_6_elsa_data_v2.tab",
            nurse_data_path="data/raw/wave_6_elsa_nurse_data_v2.tab"):
    """
    Loads core data and nurse visit data from wave 6 and merges them into a single pandas dataframe.
    :param core_data_path: {string} path to Elsa wave 6 core raw data
    :param nurse_data_path: {string} path to Elsa wave 6 nurse visit raw data
    :return: merged {pandas DataFrame}, merged dataframe on the identification column 'idauniq'
    """

    nurse = pd.read_csv(nurse_data_path, sep='\t', lineterminator='\n', header=(0))
    main = pd.read_csv(core_data_path, sep='\t', lineterminator='\n', header=(0), low_memory=False)
    merged = main.merge(right=nurse, on='idauniq')
    return merged


def frailty_level_w6(sex, height, weight, grip_strength, walking_time, exhaustion, activity_level):
    """
    Calculates Fried's Frailty Phenotype (FFP) starting from wave 6 ELSA merged database.

    :param sex: {int} 1 = Male, 2 = Female (indsex from core data)
    :param height: {float} measure in cm (height from core data)
    :param weight: {float} measure in kg (weight from nurse visit)
    :param grip_strength: {float} measure with dominant hand in kg (max between mmgsd1, mmgsd2 and mmgsd3 from nurse visit)
    :param walking_time: {float} measure in m/s (min between MMWlkA and MMWlkB from core data)
    :param activity level: {int} from 3 (max activity) to 12 (sum of HeActa HeActb and HeActc from core data)
    :param exhaustion:{int}  from 2 (exhausted) to 4 (sum of PScedB and PScedH from core data)
    :return: {int} -1 if unable to calculate; 1 if not frail; 2 if pre-frail; 3 if frail.
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
    Adds Fried's Frailty Phenotype ('FFP') column to wave 6 ELSA merged database.

    :param elsa_w6_merged: {pandas DataFrame} ELSA wave 6 with core data and nurse visit merged
    :param drop_columns: {Bool} if True deletes the columns used for FFP computation, default=False
    :param drop_rows: {Bool} if True deletes the rows for which FFP could not be calculated, default=False
    :return: elsa_w6_merged {pandas DataFrame} original dataframe with FFP column added
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
                                                      'mmgsd1', 'mmgsd2', 'mmgsd3',
                                                      'MMWlkA', 'MMWlkB',
                                                      'PScedB', 'PScedH',
                                                      'HeActa', 'HeActb', 'HeActc'])

    if drop_rows:
        elsa_w6_merged = elsa_w6_merged.loc[elsa_w6_merged['FFP'] > 0]

    return elsa_w6_merged


def group_pre_frailty(frailty_dataframe, frailty_variable="FFP"):
    """
    In a dataframe with the FFP (target) variable, groups frail and pre-frail subjects under the same value

    :param frailty_dataframe: {pandas DataFrame} with a frailty column (1=non-frail, 2=pre-frail, 3=frail)
    :param frailty_variable: {string} frailty variable (1=non-frail, 2=pre-frail, 3=frail), default=FFP
    :return: {pandas DataFrame} with binary FFP column (1=non-frail, 2=pre-frail + frail) instead of multiclass
    """
    result = [min(frailty_dataframe.loc[i, frailty_variable], 2) for i in frailty_dataframe.index]
    frailty_dataframe[frailty_variable] = np.array(result)
    return frailty_dataframe


def replace_missing_values_w6(frailty_dataframe, regex_list=None, replace_negatives=True, replace_nan=True):
    """
    Replaces nan entries, str entries and negative floats with 0

    :param frailty_dataframe: {pandas DataFrame} with frailty column FFP
    :param regex_list: {list of strings} list of regular expressions to replace, default tailored to merged w6 ELSA
    :param replace_negatives: {Boolean}, whether to replace negative values with 0 or not, default=True
    :param replace_nan: {Boolean} whether to replace NaN and None values with 0 or not, default=True
    :return: {pandas DataFrame} with zeroes instead of strings and negatives
    """

    if regex_list is None:
        regex_list = ["\s",
                      ".*:.*:.*", ".*-.*-.*",
                      # "A.*", "B.*", "C.*", "D.*", "E.*", "N.*", "P.*", "R.*", "S.*", "T.*", "V.*",
                      "[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz]",
                      ]
    frailty_dataframe.replace(to_replace=regex_list, value=0, inplace=True, regex=True)
    frailty_dataframe = frailty_dataframe.astype(float)
    if replace_negatives:
        frailty_dataframe.replace(to_replace=[i for i in range(-1, -10, -1)] + [str(i) for i in range(-1, -10, -1)],
                                  value=0, inplace=True)
    if replace_nan:
        frailty_dataframe.fillna(0, inplace=True)
        frailty_dataframe.replace(to_replace=np.nan, value=0, inplace=True)
    return frailty_dataframe


def remove_constant_features(X):
    """
    Removes constant features from the dataframe

    :param X: {pandas DataFrame}
    :return: {pandas DataFrame}
    """

    return X.loc[:, X.apply(pd.Series.nunique) != 1]


def min_max_scaling(X):
    """
    Scales values of the dataframe between 0 and 1 with min-max logic, except for the target frailty variable

    :param X: {pandas DataFrame} df to scale
    :return: {pandas DataFrame} scaled following min-max logic
    """

    X_scaled = (X - X.min()) / (X.max() - X.min())
    X_scaled.fillna(0, inplace=True)
    return X_scaled


def separate_target_variable(df, target_variable="FFP"):
    """
    Separates target variable column from the rest of the df.

    :param df: {pandas DataFrame} df containing target variable
    :param target_variable: {string} name of the target variable column within df
    :return: X, y
    """

    variables = list(df.columns.values)
    variables.remove(target_variable)
    X = df.loc[:, variables]
    y = df.loc[:, target_variable]
    return X, y


def feature_selection_list(X, y, score_func=chi2, k=100):
    """
    Selects k best features from the frailty dataframe using a statistical score function

    :param X: {pandas DataFrame} (n_samples, n_features)
    :param y: {numpy array} (n_samples) frailty target variable
    :param score_func: {string or function} statistical function to use to rate the k-best variables, default=chi2
    :param k: {int} number of variables to select, default=100
    :return: {list} with the names of the k-best variables
    """

    if type(score_func) == str:
        score_func = function_dict[score_func]
    select = SelectKBest(score_func=score_func, k=k)
    select.fit_transform(X, y)
    cols_idxs = np.array(select.get_support(indices=True))
    return cols_idxs.tolist()


def feature_selection_df(X, y, score_func=chi2, k=100):
    """
    Selects k variables from dataset X based on a score_func. Uses feature_selection_list() logic.

    :param X: {pandas DataFrame} (n_samples, n_features)
    :param y: {numpy array} (n_samples) frailty target variable
    :param score_func: {string or function} statistical function to use to rate the k-best variables, default=chi2
    :param k: {int} number of variables to select, default=100
    :return: {pandas DataFrame} (n_samples, k) filtered dataframe containing only k best variables
    """

    if type(score_func) == str:
        score_func = function_dict[score_func]
    selected_columns = feature_selection_list(X=X, y=y, score_func=score_func, k=k)
    return X.iloc[:, selected_columns]


def iterate_mutual_info_selection(x, y, score_func=mutual_info_classif,
                                  initial_k=100, iterations=10, final_k=50):
    """
    Iterates mutual_info_classif feature selection to keep the most consistent selections throughout the iterations

    :param x:
    :param y:
    :param score_func:
    :param initial_k:
    :param iterations:
    :param final_k:
    :return:
    """
    if type(score_func) == str:
        score_func = function_dict[score_func]
    selected_variables = []
    for i in range(iterations):
        selected_variables += feature_selection_list(X=x, y=y, k=initial_k, score_func=score_func)
    count = dict.fromkeys(selected_variables)
    for var in count.keys():
        count[var] = selected_variables.count(var)
    sorted_count = dict(sorted(count.items(), key=lambda x: x[1], reverse=True))
    final_variables = list(sorted_count.keys())[:final_k]
    best_features_dataframe = x.iloc[:, final_variables]
    # best_features_dataframe[frailty_variable] = y
    return best_features_dataframe


def joint_feature_selection_df(X, y, score_functions=[chi2, f_classif, mutual_info_classif], k=100):
    """
    Performs feature selection with different score function and then joins the obtained dataframes.

    :param X: {pandas DataFrame} (n_samples, n_features)
    :param y: {numpy array} (n_samples) frailty target variable
    :param score_functions: {list of string or function} statistical functions to use to rate the k-best variables
    :param k: {int} number of variables to select at each iteration, default=100
    :return: {pandas DataFrame} (n_samples, k <= n_features <= k * len(score_functions)) filtered dataframe containing
            the union set of the best variables
    """

    selected_variables = []
    for function in score_functions:
        selected_variables += feature_selection_list(X=X, y=y, score_func=function, k=k)
    selected_variables = list(set(selected_variables))
    return X.iloc[:, selected_variables]


def save_selected_df(selected_X, y, frailty_column_name="FFP", function="", k="", iterations="", folder="data/best_features/", wave=""):
    """
    Save a .tab file after variable selection

    :param selected_X: {pandas DataFrame} (n_samples, n_features) dataframe to save as a .tab file
    :param function: {string} statistical function used for feature selection, default=""
    :param k: {string o int} number of selected variables, default=""
    :param folder: {string} path to the folder where the dataframe should be saved
    :return: selected_frailty_dataframe {pandas DataFrame} the same dataframe given as input
    """

    saving_X = selected_X.copy()
    saving_X[frailty_column_name] = y
    if iterations == "":
        selected_X.to_csv(
            folder + str(wave) + "_frailty_selected_" + str(k) + "_features_" + str(function) + ".tab",
            sep='\t', index=False, quoting=3, escapechar='\\')
    else:
        selected_X.to_csv(
            folder + str(wave) + "_frailty_selected_" + str(k) + "_features_" + str(function) + "_" + str(iterations) + "_iterations" + ".tab",
            sep='\t', index=False, quoting=3, escapechar='\\')
    return selected_X


if __name__ == '__main__':
    print("Starting process...")

    fried = pd.read_csv(filepath_or_buffer='data/raw/wave_6_frailty_FFP_data.tab', sep='\t', lineterminator='\n',
                        header=0, low_memory=False)
    print(fried.shape)
    frailty_variable = "FFP"
    # Preprocess data
    fried = replace_missing_values_w6(frailty_dataframe=fried, replace_nan=True, replace_negatives=True)
    fried = remove_constant_features(fried)
    # Grouping frailty with pre-frailty to make the dataset balanced
    fried = group_pre_frailty(frailty_dataframe=fried, frailty_variable=frailty_variable)
    print("Non-frail: " + str(fried.loc[fried[frailty_variable] == 1].shape))
    print("frail: " + str(fried.loc[fried[frailty_variable] == 2].shape))
    X, y = separate_target_variable(df=fried, target_variable=frailty_variable)
    X = min_max_scaling(X=X)
    # Feature selection
    k = 30
    selection_functions = ["chi2", "f_classif", "mutual_info_classif"]
    selected_X = joint_feature_selection_df(X=X, y=y, score_functions=selection_functions, k=k)
    save_selected_df(selected_X=selected_X, y=y, frailty_column_name=frailty_variable,
                     function="chi2+f_classif+mutual_info_classif", k=selected_X.shape[1],
                     folder="data/best_features/", wave="w6")
    print(selected_X.shape)
    print(selected_X.columns)

    # model = SVR(kernel="linear")
    # # model = LogisticRegression()
    # selector = RFE(estimator=model, n_features_to_select=k, step=1)
    # selector = selector.fit(X, y)
    # selected_X = X.iloc[:, (selector.support_)]
    print("All done!")
