# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# filter the rows only for this catcher
df = pd.read_csv('./data/2020-train.csv')
catcher_f06c9fdf = np.array(df['catcher_id'] == 'f06c9fdf')
df = df[catcher_f06c9fdf]

df_test = pd.read_csv('./data/2020-test.csv')
catcher_f06c9fdf_test = np.array(df_test['catcher_id'] == 'f06c9fdf')
df_test = df_test[catcher_f06c9fdf_test]

pitch_id = set(df.iloc[:, 35])
pitch_id_test = set(df_test.iloc[:, 35])
for x in pitch_id_test:
    if x not in pitch_id:
        pitch_id.add(x)

# from the data available, the catchers is involved in 10998 different pitches.


# some data are skewed
# for ex, pitcher_side has some cells with R, L
# instead of right, left
def fix_skewed_data(df):
    df['pitcher_side'] = np.where(df.pitcher_side == 'R', 'Right', df.pitcher_side)
    df['pitcher_side'] = np.where(df.pitcher_side == 'L', 'Left', df.pitcher_side)
    df['batter_side'] = np.where(df.batter_side == 'R', 'Right', df.batter_side)
    df['batter_side'] = np.where(df.batter_side == 'L', 'Left', df.batter_side)
    return df

df = fix_skewed_data(df)
df_test = fix_skewed_data(df_test)

def left_right(df):
    pitcher_right = np.array(df['pitcher_side'] == 'Right')
    pitcher_left = np.array(df['pitcher_side'] == 'Left')

    pR = df[pitcher_right]
    pL = df[pitcher_left]

    return [pR, pL]

df_RL = left_right(df)
df_RL_test = left_right(df_test)

pR, pL = left_right(df)
pR_test, pL_test = left_right(df_test)

right_total = len(pR) + len(pR_test)
left_total = len(pL) + len(pL_test)

pR.groupby(['pitch_call']).size()
pL.groupby(['pitch_call']).size()

pR.groupby(['pitch_type']).size()
pL.groupby(['pitch_type']).size()