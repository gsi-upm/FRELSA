import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models import get_metrics, get_classifiers_metrics
from preprocess import add_fried_w6, load_w6, load_w5, alternated_merge_for_lstm, save_lstm_data, load_lstm_data, \
    replace_missing_values_w6, remove_constant_features, group_pre_frailty, min_max_scaling

if __name__ == '__main__':
    print("Initializing...")

    # core_data_path = "data/raw/wave_6_elsa_data_v2.tab"
    # nurse_data_path = "data/raw/wave_6_elsa_nurse_data_v2.tab"
    # fried = add_fried_w6(elsa_w6_merged=load_w6(core_data_path=core_data_path, nurse_data_path=nurse_data_path),
    #                      drop_columns=True, drop_rows=True)
    # fried.to_csv('data/raw/wave_6_frailty_FFP_data.tab', sep='\t', index_label='idauniq', quoting=3, escapechar='\\')


    # bf_w6 = pd.read_csv("data/best_features/w6_frailty_selected_83_features_chi2+f_classif+mutual_info_classif.tab",
    #                     sep='\t', lineterminator='\n', index_col='idauniq', header=(0))
    # df_w5 = load_w5(core_data_path="data/raw/wave_5_elsa_data_v4.tab", acceptable_features=np.array(bf_w6.columns),
    #                 acceptable_idauniq=np.array(bf_w6.index))
    # print(df_w5.shape)
    # X, y = alternated_merge_for_lstm(older_df=df_w5, frailty_df=bf_w6)
    # X.to_csv("data/best_features/w5+w6_frailty_selected_22_features_chi2+f_classif+mutual_info_classif.tab",
    #              sep='\t', index_label='idauniq', quoting=3, escapechar='\\')
    # save_lstm_data(alternated_merged_df=X, y=y, target_variable="FFP", folder_path="data/best_features",
    #                filename="TEST_w5+w6_frailty_selected_22_features_chi2+f_classif+mutual_info_classif.tab")
    # X, y = load_lstm_data(filepath="data/best_features/TEST_w5+w6_frailty_selected_22_features_chi2+f_classif+mutual_info_classif.tab")

    random_state = 25
    frailty_variable = "FFP"
    data_file = "wave_5_elsa_data_v4.tab"
    df_w6 = pd.read_csv("data/raw/wave_6_frailty_FFP_data.tab",
                        sep='\t', lineterminator='\n', index_col='idauniq', header=(0), low_memory=False)
    df_w5 = load_w5(core_data_path="data/raw/" + str(data_file), index_col='idauniq', acceptable_features=None,
                    acceptable_idauniq=np.array(df_w6.index))
    df_w6 = df_w6.loc[df_w5.index]
    df_w6.sort_index(inplace=True)
    X = df_w5.sort_index()
    y = df_w6.loc[:, frailty_variable]
    # Group frailty and pre-frailty to make dataset balanced
    y = [min(i, 2) for i in y]
    print(min(y))
    print(max(y))
    print(y)
    X = replace_missing_values_w6(frailty_dataframe=X, replace_nan=True, replace_negatives=True)
    X = remove_constant_features(X)
    X = min_max_scaling(X=X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    metrics_df = get_classifiers_metrics(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, random_state=random_state)
    metrics_df.to_csv("data/metrics/wave_5/metrics_of_" + data_file.split('.', 1)[0] + '_random_state_' + str(random_state) + '.tab',
                      sep='\t', index_label='idauniq', quoting=3, escapechar='\\')


