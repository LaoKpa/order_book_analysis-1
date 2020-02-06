import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

#Constants
PROBA_THRESH = 0.75

'''
Reads from the .csv
'''
def download_from_csv(num_of_rows, skiprows=0):    
    df_to_analyze = pd.read_csv(
        '/home/quotermin/ml/trading/candles_ticks_orderbook/SBER_data.csv', 
        header = 0,
        index_col = 0,
        nrows = num_of_rows,
        skiprows=skiprows
    )
    return df_to_analyze


'''
Appends columns with target variable
as max distance to low and high during
time_range
'''
def append_distance_per_range(df_to_analyze, range_to_look_forward):
    
    #global df_to_analyze
    
    df_to_analyze['dist_to_max_per_range'] = np.array(df_to_analyze[['close']]\
        .iloc[::-1].rolling(range_to_look_forward, min_periods=1).max().iloc[::-1])\
        - np.array(df_to_analyze[['close']])

    df_to_analyze['dist_to_min_per_range'] = np.array(df_to_analyze[['close']])\
        - np.array(df_to_analyze[['close']]\
        .iloc[::-1].rolling(range_to_look_forward, min_periods=1).min().iloc[::-1])
	
	#return df_to_analyze
	

'''
Creates column to indicate movement above and below
median movement of the price as the target variable
'''
def append_up_or_down(df_to_analyze, first_bound, second_bound):
    
    #global df_to_analyze
    
    conditions = [
        np.logical_and(
            df_to_analyze['dist_to_max_per_range'] > np.percentile(
                df_to_analyze['dist_to_max_per_range'], first_bound
            ),
            df_to_analyze['dist_to_min_per_range'] < np.percentile(
                df_to_analyze['dist_to_min_per_range'], second_bound
            )
        ),
        np.logical_and(
            df_to_analyze['dist_to_max_per_range'] < np.percentile(
                df_to_analyze['dist_to_max_per_range'], second_bound
            ),
            df_to_analyze['dist_to_min_per_range'] > np.percentile(
                df_to_analyze['dist_to_min_per_range'], first_bound
            )
        )
    ]

    choices = ['up', 'down']
    df_to_analyze['y'] = np.select(conditions, choices, default='nothing')
    #df_to_analyze.y=df_to_analyze.y.shift(-1)
    df_to_analyze = df_to_analyze.dropna()
	
	
'''
Splits the data into features and targets
and further splits it into train and test
'''
def split_the_data(df_to_analyze):
    
    #global df_to_analyze
    
    #!!!DROPS TOO MANY ROWS!!!
    df_to_analyze = df_to_analyze.replace([np.inf, -np.inf], np.nan).dropna()

    X = df_to_analyze.drop(['dist_to_max_per_range', 'dist_to_min_per_range', 'y'], axis=1)
    y = df_to_analyze.y

    #Creates the oldest data as the train set and the newest as the test set
    train_size = int(df_to_analyze.shape[0] * 0.75)
    X_train = X.iloc[:train_size, :]
    y_train = y[:train_size]
    X_test = X.iloc[train_size:, :]
    y_test = y.iloc[train_size:]
    return X_train, y_train, X_test, y_test


'''
Creates the model, fits it,
makes predictions
'''
def fit_the_model(X_train, y_train, X_test, y_test):
    
    #global X_train, y_train, X_test, y_test
    
    clf_rf = RandomForestClassifier(
        n_estimators = 300 ,
        max_depth = 9,
        min_samples_split = 3,
        min_samples_leaf = 2,
        n_jobs = -1
    )

    clf_rf.fit(X_train, y_train)
    
    y_pred_proba = clf_rf.predict_proba(X_test)

    # Creates an empty 1D string array and fills it with default string
    y_pred = np.empty(len(y_pred_proba), dtype=object, order='C')
    y_pred[:] = 'nothing'

    # Fills the array with predictions according threshold
    y_pred[np.where(y_pred_proba[:, 0] >= PROBA_THRESH)] = 'down'
    y_pred[np.where(y_pred_proba[:, 2] >= PROBA_THRESH)] = 'up'
    
    report = classification_report(
        y_test, y_pred, output_dict = True
    )
    
    print('Clf')
    if 'up' in report:
        print(
            'Precision up:', report['up']['precision']
        )
    if 'nothing' in report:
        print(
            'Precision nothing:', report['nothing']['precision']
        )
    if 'down' in report:
        print(
            'Precision down:', report['down']['precision']
        )
    
    return clf_rf


'''
FUNCTION
Selects most important features
from the previous model, creates new one,
fits it, makes predictions
'''

def fit_the_model_selected(clf, X_train, y_train, X_test, y_test):
    sel = SelectFromModel(clf)
    sel.fit(X_train, y_train)

    X_important_train = sel.transform(X_train)
    X_important_test = sel.transform(X_test)

    clf_important = RandomForestClassifier(
        n_estimators = 9,
        max_depth = 9,
        min_samples_split = 3,
        min_samples_leaf = 2,
        n_jobs = -1
    )

    clf_important.fit(X_important_train, y_train)
    
    y_pred_proba = clf_important.predict_proba(X_important_test)

    # Creates an empty 1D string array and fills it with default string
    y_pred = np.empty(len(y_pred_proba), dtype=object, order='C')
    y_pred[:] = 'nothing'

    # Fills the array with predictions according threshold
    y_pred[np.where(y_pred_proba[:, 0] >= PROBA_THRESH)] = 'down'
    y_pred[np.where(y_pred_proba[:, 2] >= PROBA_THRESH)] = 'up'
    
    report = classification_report(y_test, y_pred, output_dict = True)
    
    print('Clf important')
    if 'up' in report:
        print(
            'Precision up:', report['up']['precision']
        )
    if 'nothing' in report:
        print(
            'Precision nothing:', report['nothing']['precision']
        )
    if 'down' in report:
        print(
            'Precision down:', report['down']['precision']
        )
    
    return clf_important, X_train.columns[sel.get_support()]



if __name__ == '__main__':
	run()