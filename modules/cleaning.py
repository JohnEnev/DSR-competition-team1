import pandas as pd

def merge(df_train, df_store):
    '''
    Merge train data with store data based on
    values in Stores column
    '''
    full_df = df_train.merge(df_store, how='left', on='Store')
    return full_df

def drop_column(df, column='Customers'):
    '''
    Remove the Customers column as instructed by competition instructors.
    '''
    df_new = df.drop(str(column), axis=1)
    return df_new

def drop_null_targets(df, target='Sales'):
    '''
    Remove the null targets (i.e. Sales)
    '''
    null_target = df[df[str(target)].isnull()]
    df_new = df[~df.index.isin(null_target.index)]
    return df_new

def count_null_features(df):
    '''
    Count how many null values are in each column
    '''
    tot_rows = df.shape[0]
    for column in df.columns:
        null_rows = df.loc[df.loc[:, str(column)].isnull()].shape[0]
        frac = null_rows / tot_rows
        print(f"Column {str(column)} has {frac * 100 : .0f}% null values.")


def drop_null_features(df, threshold=0.03, verbose=False):
    '''
    Remove the null features from a column if number of rows with null
    values are very small, i.e. less than a certain percentage of the
    full data rows.
    The default threshold is 3%.
    '''
    tot_rows = df.shape[0]
    if verbose:
        print("Total number of rows in full data set: ", tot_rows)
    for column in df.columns:
        null_rows = df.loc[df.loc[:, str(column)].isnull()].shape[0]
        frac = null_rows / tot_rows
        if verbose:
            print(f"Column {str(column)} has {frac * 100 : .0f}% null values.")
        if frac > 0 and frac <= threshold:
            df = df.loc[~df.loc[:, str(column)].isnull()]
            if verbose:
                print(f"Removed rows with null value from {str(column)}.")
    if verbose:
        print("Total number of rows in clean data set: ", df.shape[0])
    return df

def rough_cleaning(df, threshold=0.1, verbose=False):
    '''
    Remove the null features from a column if number of rows with null
    feature is less than a certain percentage of the full data rows
    (i.e. the threshold), drop the whole column if number of rows with
    null feature is larger than the threshold.
    The default threshold is 10%.
    '''
    tot_rows = df.shape[0]
    if verbose:
        print("Total number of rows in full data set: ", tot_rows)
    for column in df.columns:
        null_rows = df.loc[df.loc[:, str(column)].isnull()].shape[0]
        frac = null_rows / tot_rows
        if verbose:
            print(f"Column {str(column)} has {frac * 100 : .0f}% null values.")
        if frac > 0 and frac <= threshold:
            df = df.loc[~df.loc[:, str(column)].isnull()]
            if verbose:
                print(f"Removed rows with null value from {str(column)}.")
        elif frac > threshold:
            df = df.drop(str(column), axis=1)
            if verbose:
                print(f"Dropped the column {str(column)}.")
    if verbose:
        print("Total number of rows in clean data set: ", df.shape[0])
    return df

