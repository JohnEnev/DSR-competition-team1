import pandas as pd
import numpy as np
import datetime as dt



def one_hot_encoding(df, column):
    '''
    Function to take a column and transform and one-hot encode it.
    Returns a dataframe with the original column removed and the new encoded columns added.
    '''
    # create dummies series
    dummies = pd.get_dummies(df.loc[:, column], prefix=str(column + ' '))
    # concat original dataframe and dummies
    df_new = pd.concat([df, dummies], axis=1)
    # remove the original column
    df_new = df_new.drop(column, axis=1)

    return df_new


def dates_features(df):
    '''
    Function to create new date features and delete DayOfWeek.
    Used before train/test split.
    Return a new dataframe.
    '''
    df_new = df.copy()
    #Transform Date column into Datetime object
    df_new.Date = pd.to_datetime(df_new.Date)
    #Delete DayOfWeek
    df_new = df_new.drop('DayOfWeek', axis=1)
    #Create new features
    df_new['month'] = df_new.Date.dt.month
    df_new['day_of_week'] = df_new.Date.dt.dayofweek
    #Monday is 0
    df_new['day_of_month'] = df_new.Date.dt.day
    df_new['is_monday'] = np.where(df_new['day_of_week'] == 0, 1, 0)
    df_new['is_saturday'] = np.where(df_new['day_of_week'] == 5, 1, 0)

    return df_new