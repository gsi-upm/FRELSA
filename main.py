import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


if __name__ == '__main__':
    print("Initializing...")
    df = pd.read_csv("data/wave_6_frailty_FFP_data.tab", sep='\t', lineterminator='\n', header=(0), low_memory=False)
    print(df.shape, df.columns.values[-1])
