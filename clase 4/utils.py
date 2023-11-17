import numpy as np
import pandas as pd



def drop_duplicates_features(df:pd.DataFrame,verbose = False)->dict:
    duplicated_feat_groups = {}
    # create an empty list to collect features
    # that are found to be duplicated
    _duplicated_feat = []
    # iterate over every feature in our dataset:
    for i in range(0, len(df.columns)):

        # choose 1 feature:
        feat_1 = df.columns[i]

        # check if this feature has already been identified
        # as a duplicate of another one.

        # If this feature was already identified as a duplicate, we skip it, if
        # it has not yet been identified as a duplicate, then we proceed:
        if feat_1 not in _duplicated_feat:

            # create an empty list as an entry for this feature in the dictionary:
            duplicated_feat_groups[feat_1] = []

            # now, iterate over the remaining features of the dataset:
            for feat_2 in df.columns[i + 1:]:

                # check if this second feature is identical to the first one
                if df[feat_1].equals(df[feat_2]):

                # if it is identical, append it to the list in the dictionary
                    duplicated_feat_groups[feat_1].append(feat_2)

                    # and append it to our list for duplicated variables
                    _duplicated_feat.append(feat_2)

    df = df[duplicated_feat_groups.keys()]
    if verbose:
        print(f'Columns duplicated dropped:\n{_duplicated_feat}')
    return df





def drop_columns_null_values_according_threshold(df:pd.DataFrame, threshold:float,verbose = False)->pd.DataFrame:
    
    df_missing = df.isna().sum().to_frame().reset_index()
    df_missing.columns = ['feature_name','# nulls']
    df_missing['proportion_null'] = df_missing['# nulls'] / len(df)
    df_missing.sort_values(by = ['proportion_null'], inplace = True, ascending = False)
    columns_more_than_threshold_nulls = df_missing[df_missing['proportion_null'] > threshold]['feature_name'].to_list()

    # Drop the columns with > 70% null of the dataframe
    if verbose:
        print(f'Columns with more than {threshold*100}% missing values:\n{columns_more_than_threshold_nulls}')
        
    columns_more_than_threshold_nulls = [ i for i in df.columns if i not in columns_more_than_threshold_nulls ]
    df = df[columns_more_than_threshold_nulls]
    

    
    return df





def get_rid_constant_features(df:pd.DataFrame,verbose = False)->pd.DataFrame:
    
    constant_features = [feat for feat in df.columns if df[feat].nunique() == 1]
    df = df[[i for i in df.columns if i not in constant_features]]
    if verbose:
        print(f'Columns constants dropped:\n{constant_features}')
    return df