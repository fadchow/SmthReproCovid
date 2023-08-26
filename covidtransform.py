import pandas as pd
import numpy as np

def load_and_clean(nama_file, region_filter, type_of_covid):
    df = pd.read_csv(nama_file)
    melted_df = pd.melt(df, id_vars=df.iloc[:,:4], var_name='date', value_name=f'{type_of_covid} count')
    melted_df['date'] = pd.to_datetime(melted_df['date'],format='mixed')
    country_df = melted_df[melted_df['Country/Region'].str.contains(region_filter, case=False)]
    return country_df

def convert_to_int(x):
    if pd.isna(x):
        return 0
    return int(x)

def calculate_I_t(infected, I_t_before,gamma): #
    if pd.isna(infected):
        return I_t_before
    if I_t_before == -1:
        return infected
    I_t = ((1 - gamma) * I_t_before) + infected
    return I_t

def gr(I_t, I_t_before): #
    if I_t_before == 0:
        return 0  # prevent division by 0
    if pd.isna(I_t) or pd.isna(I_t_before):
        return np.nan
    return (I_t - I_t_before) / I_t_before

def getconfirmed(df: pd.DataFrame):
    df = df.iloc[:,4:] # or -> df = df.iloc[:, [4, 5]]
    return reindex(df)

def new_case_per_day(df: pd.DataFrame):  # calculate new case per day
    df['New_Cases_per_day'] = df['confirmed count'].diff()
    df['New_Cases_per_day'] = df['New_Cases_per_day'].apply(convert_to_int)
    return df

def infected_per_day(df: pd.DataFrame, gamma: float=1/7): # calculate Infected data
    df['I_t'] = 0
    for index, row in df.iterrows():
        if index == 0:
            df.at[index, 'I_t'] = calculate_I_t(row['New_Cases_per_day'], -1,gamma)
        else:
            df.at[index, 'I_t'] = calculate_I_t(row['New_Cases_per_day'], df.at[index - 1, 'I_t'], gamma)

    df['I_t'] = df['I_t'].apply(lambda x: round(x,2))
    return df

def growth(df: pd.DataFrame): # calculate Growth data
    df['gr'] = df.apply(lambda x: gr(x['I_t'], df.loc[x.name - 1, 'I_t']) if x.name > 0 else 0, axis=1)
    return df

def rrate(df: pd.DataFrame, gamma: float=1/7): # calculate Reproduction Rate data
    df['R_k'] = df['gr'].apply(lambda x: (x/gamma)+1)
    return df

def bydate(df: pd.DataFrame, first_date: str='2020-01-22', last_date: str='2023-03-09'): # filter data by date
    df = df[(last_date >= df['date']) & (first_date <= df['date'])]
    return reindex(df)

def sum_confirmed(df: pd.DataFrame, sum: int=100): # filter data with minimum cumulative case
    index_toSum = 0
    for index in range(len(df)):
        if df.at[index, 'confirmed count'] >= sum:
            index_toSum = index
            break
    df = df[df.index > index_toSum]
    return reindex(df)

def reindex(df: pd.DataFrame): # Re-indexing data
    df = df.reset_index()
    return df.drop(columns = ['index'])