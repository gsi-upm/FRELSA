import numpy as np
import pandas as pd
from CFSmethod import CFS
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFE

pd.options.mode.chained_assignment = None

function_dict = {
    "chi2": chi2,
    "f_classif": f_classif,
    "mutual_info_classif": mutual_info_classif,
}

variables_dict = {

    'hemobwa': "Mobility: difficulty walking 100 yards",
    'hemobch': "Mobility: difficulty getting up from chair after sitting long periods",
    'hemobcs': "Mobility: difficulty climbing several flights stairs without resting",
    'hemobcl': "Mobility: difficulty climbing one flight stairs without resting",
    'hemobst': "Mobility: difficulty stooping, kneeling or crouching",
    'hemobpu': "Mobility: difficulty pulling or pushing large objects",
    'hemobli': "Mobility: difficulty lifting or carrying weights over 10 pounds",
    'hemob96': "Mobility: whether said had none of listed difficulties",
    'headldr': "ADL: difficulty dressing, including putting on shoes and socks",
    'SPCar': "Whether respondent has use of car or van when needed, as a driver or a passenger",
    'headlba': "ADL: difficulty bathing or showering",
    'SPCarB': "Whether respondent drove a car or van themself in the past?",
    'headlsh': "IADL: difficulty shopping for groceries",
    'SPTraB6': "Reason respondent does not use public transport: their health prevents them",
    'SPTraB12': "Reason respondent does not use public transport: difficulties with mobility",
    'headlho': "[Difficulty] Doing work around the house or garden",
    'heaidca': "Aids used: cane or walking stick",
    'headl96': "ADL&IADL: whether said had none of listed difficulties",
    'PSAgF': "Self-perceived age",
    'NumMeds': "DRUG: NumMeds",
    'SPTraM6': "Merged var - reason resp does not use public transport: health prevents them",
    'heaid96': "Aids used: not use any of listed aids (walking or special eating or personal alarm)",
    'hecach': "Aids: whether cane acquired since last interview",
    'SPTraM12': "Merged var - reason respondent does not use public transport: mobility problems",
    'hecacov': "Aids, cane or walking stick: whether covered all costs",
    'scinp1': "Places respondent has used the internet or email in last 3 months: At home",
    'scina02': "Activities respondent used internet for in last 3 months: Finding information about goods and services",
    'scina05': "Activities respondent used internet for in last 3 months: Shopping/buying goods or services",
    'mmlsre': "LEGRAISE: Leg raise (eyes shut): Outcome",
    'mmlssc': "LEGRAISE: Leg raise (eyes shut): Respondent feels safe to attempt it",
    'mmlore': "LEGRAISE: Leg raise (eyes open): Outcome",
    'CaTkA': "Received help in last month: walking 100 yards",
    'CaTkB': "Received help in last month: climbing several flights of stairs without resting",
    'CaTkC': "Received help in last month: climbing one flight of stairs without resting",
    'CaTkD': "Received help in last month: dressing, including putting on shoes and socks",
    'CaTkE': "Received help in last month: walking across a room",
    'CaTkF': "Received help in last month: bathing or showering",
    'CaTkG': "Received help in last month: eating, such as cutting up food",
    'CaTkH': "Received help in last month: getting in and out of bed",
    'CaTkI': "Received help in last month: using the toilet, including getting up or down",
    'CaTkJ': "Received help in last month: shopping for groceries",
    'CaTkK': "Received help in last month: taking medications",
    'CaTkL': "Received help in last month: doing work around the house or garden",
    'CaTkM': "Received help in last month: managing money, such as paying bills and keeping track of expenses",
    'CaTNo': "COMPUTED: Number of activities respondent has received help with in the last month",
    'CaWIn': "COMPUTED: Received eligible help and asked questions about it",
    'mmrrre': "CHAIRRAISE: Outcome of multiple chair rises (number of rises completed)",
    'scqola': "CASP19 scale: How often feels age prevents them from doing things they like",
    'scqolh': "CASP19 scale: How often feels their health stops them doing what they want to do",
    'scqolo': "CASP19 scale: How often feels full of energy these days",
    'htfvc': "LUNG: Highest technically satisfactory value for FVC",
    'htfev': "LUNG: Highest technically satisfactory value for FEV",
    'Hehelf': "Self-reported general health",
    'HeLWk': "Whether has self-reported health problem/disability that limits paid work",
    'Hetemp': "Whether expects health problem/disability to last less than 3 months",
    'HeFunc': "Difficulty walking 1/4 mile unaided",
    'MMRROC': "(D) Chair rise: Outcome of multiple chair rises, split by age",
    'MMFTRE2': "(D) Outcome of full tandem stand according to age",
    'Indobyr': "Definitive year of birth collapsed at 90 plus",
    'indager': "Definitive age variable collapsed at 90+ to avoid disclosure",
    'CaHFoNo1': "Formal help received - with moving (catka, b, c, e, h, i): None of these",
    'CahFMNo1': "Formal help received - with moving (catka, b, c, e, h, i): None of these (merged var)",
    'CaHInHW4': "Informal help received - with shopping & work around the house (catkaj, L): Husb/Wife/Partner",
    'CaHFoNo4': "Formal help received - with shopping & work around the house (catkaj, L): None of these",
    'CahFMNo4': "Formal help received - with shopping & work around the house (catkaj, L): None of these (merged var)",
    'CaHpC': "How often all help received meets needs",
    'Indobyr_x': "Definitive year of birth collapsed at 90 plus",
    'CaFnd1': "Type of provider for questions about care patterns (provider 1)",
    'CaHpF': "How often does provider 1 care patterns (CaFnd1) help?",
    'CaHpW': "Does provider 1 care patterns (CaFnd1) help during day or night?",
    'CaDP3': "How LA is involved in paying for care: no direct payment or LA management of money",
    'CaPB': "Whether has a personal budget from LA",
    'CaInA': "Whether had an income assessment by LA",
    'MedBIA3': "DRUG: Have they taken/used MedBI in the last 7 days?",
    'MedBIA4': "DRUG: Have they taken/used MedBI in the last 7 days?",
    'MedBIA5': "DRUG: Have they taken/used MedBI in the last 7 days?",
    'MedBIA7': "DRUG: Have they taken/used MedBI in the last 7 days?",
    'CaRA': "Care arranged by local authority (council care package): ask route A payments (CaPay etc)",
    'mmrrfti': "CHAIRRAISE: Time to complete 5 rises (seconds)",
    'mmrrtti': "CHAIRRAISE: Time to complete 10 rises (seconds) - only eligible if under 70 yrs",
    'CaPAdNo2': "Does anyone else pay for the care: nobody else pays (provider at CaAsk1)",
    'HePaa': "Severity of pain most of the time",
    'HePag': "Pain: how long been bothering you (if moderate or severe)",
    'PRFEV': "LUNG: Predicted value for FEV",
    'PRPEF': "LUNG: Predicted value for PEF",
    'Helim': "Whether long-standing illness is limiting",
    'DrC4': "DRUG: Drug Code [of MedBIA4]",

    'FFP': "Fried's Frailty Phenotype",
}


def load_w6(core_data_path="data/raw/wave_6_elsa_data_v2.tab",
            nurse_data_path="data/raw/wave_6_elsa_nurse_data_v2.tab", remove_duplicates=True, index_col="idauniq"):
    """
    Loads core data and nurse visit data from wave 6 and merges them into a single pandas dataframe.
    :param index_col:
    :param remove_duplicates: {Boolean} if True, remove duplicate columns from the two dataframes
    :param core_data_path: {string} path to Elsa wave 6 core raw data
    :param nurse_data_path: {string} path to Elsa wave 6 nurse visit raw data
    :return: merged {pandas DataFrame}, merged dataframe on the identification column 'idauniq'
    """

    nurse = pd.read_csv(nurse_data_path, sep='\t', lineterminator='\n', header=(0), index_col=index_col)
    main = pd.read_csv(core_data_path, sep='\t', lineterminator='\n', header=(0), index_col=index_col, low_memory=False)
    merged = main.merge(right=nurse, on=index_col, suffixes=('', '_y'))
    if remove_duplicates:
        merged.drop(merged.filter(regex='_y$').columns, axis=1, inplace=True)
    return merged


def load_w5(core_data_path="data/raw/wave_5_elsa_data_v4.tab", acceptable_features=None, acceptable_idauniq=None):
    df = pd.read_csv(core_data_path, sep='\t', lineterminator='\n', header=(0), index_col='idauniq', low_memory=False)
    if (acceptable_features is None) and (acceptable_idauniq is None):
        return df
    if not (acceptable_features is None):
        acceptable_features = np.array(set(df.columns) & set(acceptable_features))
        df = df.loc[:, acceptable_features]
        if acceptable_idauniq is None:
            return df
    # acceptable_idauniq = np.array(set(df['idauniq']) & set(acceptable_idauniq))
    df = df.filter(items=acceptable_idauniq, axis=0)
    return df


def alternated_merge_for_lstm(older_df, frailty_df, frailty_variable="FFP", older_df_label="w5", frailty_df_label="w6"):
    """

    :param older_df:
    :param frailty_df:
    :param frailty_variable:
    :param older_df_label:
    :param frailty_df_label:
    :return:
    """
    frailty_df = frailty_df.filter(items=np.array(older_df.index), axis=0).sort_index()
    frailty_df, y = separate_target_variable(df=frailty_df, target_variable=frailty_variable)
    frailty_df = frailty_df.loc[:, older_df.columns]
    older_df = replace_missing_values_w6(frailty_dataframe=older_df, replace_nan=True, replace_negatives=True)
    older_df = remove_constant_features(older_df)
    older_df = min_max_scaling(older_df)
    frailty_df["wave"] = np.array([frailty_df_label for i in range(frailty_df.shape[0])])
    frailty_df.set_index("wave", append=True, inplace=True)
    older_df["wave"] = np.array([older_df_label for i in range(older_df.shape[0])])
    older_df.set_index("wave", append=True, inplace=True)
    final = pd.concat([older_df, frailty_df]).sort_index()
    return final, y


def save_lstm_data(alternated_merged_df, y, target_variable="FFP",
                   filename="w5+w6_frailty_selected_22_features_chi2+f_classif+mutual_info_classif.tab",
                   folder_path="best_features", sep='\t', index_label='idauniq', quoting=3, escapechar='\\'):
    """

    :param alternated_merged_df:
    :param y:
    :param target_variable:
    :param filename:
    :param folder_path:
    :param sep:
    :param index_label:
    :param quoting:
    :param escapechar:
    :return:
    """
    y = np.repeat(y, 2)
    final = alternated_merged_df.copy()
    final.reset_index(inplace=True, drop=True)
    final[target_variable] = y
    final.set_index(index_label, inplace=True, drop=True)
    final.to_csv(str(folder_path) + str(filename), sep=sep, index_label=index_label, quoting=quoting,
                 escapechar=escapechar)
    return alternated_merged_df, y


def load_lstm_data(filepath="data/best_features/w5+w6_frailty_selected_22_features_chi2+f_classif+mutual_info_classif.tab",
                   index_col="idauniq", target_variable="FFP"):
    """

    :param filepath:
    :param index_col:
    :param target_variable:
    :return:
    """
    df = pd.read_csv(filepath, sep='\t', lineterminator='\n', header=(0), index_col=index_col, low_memory=False)
    y = np.array(list(df[target_variable])[1::2])
    df.drop(target_variable, axis=1, inplace=True)
    return df, y


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
                                                      'mmgsd1', 'mmgsd2', 'mmgsd3', 'mmgsn1', 'mmgsn2', 'mmgsn3',
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


def save_selected_df(selected_X, y, frailty_column_name="FFP", function="", k="", iterations="",
                     folder="data/best_features/", wave=""):
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
        saving_X.to_csv(
            folder + str(wave) + "_frailty_selected_" + str(k) + "_features_" + str(function) + ".tab",
            sep='\t', index_label='idauniq', quoting=3, escapechar='\\')
    else:
        saving_X.to_csv(
            folder + str(wave) + "_frailty_selected_" + str(k) + "_features_" + str(function) + "_" + str(
                iterations) + "_iterations" + ".tab",
            sep='\t', index_label='idauniq', quoting=3, escapechar='\\')
    return selected_X


if __name__ == '__main__':
    print("Starting process...")

    fried = pd.read_csv(filepath_or_buffer='data/raw/wave_6_frailty_FFP_data.tab', sep='\t', lineterminator='\n',
                        header=0, low_memory=False, index_col="idauniq")
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
    k = 50
    selection_functions = ["mutual_info_classif"]  # ["chi2", "f_classif", "mutual_info_classif"]
    selected_X = joint_feature_selection_df(X=X, y=y, score_functions=selection_functions, k=k)
    save_selected_df(selected_X=selected_X, y=y, frailty_column_name=frailty_variable,
                     function="_".join([str(f) for f in selection_functions]), k=selected_X.shape[1],
                     folder="data/best_features/", wave="w6")
    print(selected_X.shape)
    print(selected_X.columns)

    print("All done!")
