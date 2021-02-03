import pandas as pd

def merge(df_train, df_store):
    '''
    Merge train data with store data based on values in Stores column.
    '''
    full_df = df_train.merge(df_store, how='left', on='Store')
    return full_df

def drop_column(df, column='Customers'):
    '''
    Remove the Customers column as instructed by competition instructors.
    Returns a new dataframe.
    '''
    df_new = df.drop(str(column), axis=1)
    return df_new

def clean_column_values(df, column='StateHoliday'):
    '''
    In the StateHoliday column, transform 0.0 to 0 and all column to string.
    Returns a new dataframe.
    '''
    df_new = df.copy()
    #Remove the values where StateHoliday is NaN?
    df_new = df_new.dropna(subset=[str(column)])
    df_new[str(column)] = df_new[str(column)].apply(lambda x: str(int(x)) if x == 0.0 or x == 0 else x)
    return df_new

def drop_null_targets(df, target='Sales'):
    '''
    Remove the rows where value for Sales is null.
    Returns a new dataframe.
    '''
    null_target = df[df[str(target)].isnull()]
    df_new = df[~df.index.isin(null_target.index)]
    return df_new

def drop_zero_targets(df, target='Sales'):
    '''
    Remove the rows with zero Sales.
    Returns a new dataframe.
    '''
    zero_target = df[df[str(target)] == 0]
    df_new = df[~df.index.isin(zero_target.index)]
    return df_new

def clean_targets(df, target='Sales'):
    '''
    Drop all the rows where Sales does is null or zero.
    Returns a new dataframe.
    '''
    df_new = drop_null_targets(df, target=str(target))
    df_new = drop_zero_targets(df, target=str(target))
    return df_new

def count_null_features(df):
    '''
    Count how many null values are in each column.
    '''
    tot_rows = df.shape[0]
    for column in df.columns:
        null_rows = df.loc[df.loc[:, str(column)].isnull()].shape[0]
        frac = null_rows / tot_rows
        print(f"Column {str(column)} has {frac * 100 : .0f}% null values.")

def rough_features_cleaning(df, threshold=0.10, drop_columns=True, verbose=False):
    '''
    Remove the rows with null feature values if number of null values
    are very small.
    When the flag is true, drop the whole feature column if number of
    null values are not small.
    The default threshold is 10%.
    Transform the values in StateHoliday column.
    Returns a new dataframe.
    '''
    df_new = df.copy()
    tot_rows = df_new.shape[0]
    print("Total number of rows before cleaning: ", tot_rows)
    for column in df_new.columns:
        null_rows = df_new.loc[df_new.loc[:, str(column)].isnull()].shape[0]
        frac = null_rows / tot_rows
        if verbose:
            print(f"Column {str(column)} has {frac * 100 : .0f}% null values.")
        if frac > 0 and frac <= threshold:
            df_new = df_new.loc[~df_new.loc[:, str(column)].isnull()]
            if verbose:
                print(f"Removed rows with null value from {str(column)}.")
        if drop_columns and frac > threshold:
            df_new = df_new.drop(str(column), axis=1)
            if verbose:
                print(f"Dropped the column {str(column)}.")
    print("Total number of rows after cleaning: ", df_new.shape[0])

    df_new = clean_column_values(df_new, column='StateHoliday')

    return df_new

