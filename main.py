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
