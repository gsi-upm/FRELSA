import pandas as pd
import numpy as np

from models import get_metrics
from preprocess import add_fried_w6, load_w6, load_w5, alternated_merge_for_lstm, save_lstm_data, load_lstm_data

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

