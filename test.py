import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from finta import TA
from utils.append_indicators import append_indicators
import pickle

engine = create_engine(
	'mysql://Quotermain:Quotermain233@192.168.0.105:3306/trading_data'
)

assets = [
	'ALRS', 
	'CHMF', 
	'GAZP', 
	'GMKN', 
	'HYDR', 
	'LKOH', 
	'MGNT', 
	'MOEX', 
	'MTLR', 
	'MTSS', 
	'NVTK', 
	'ROSN', 
	'RTKM', 
	'SBER', 
	'SBERP', 
	'SIBN', 
	'SNGS', 
	'SNGSP', 
	'TATN', 
	'YNDX'
]
dict_of_tf = {
	'1_': 480, #problem
	'4_': 120,
	'15_': 32,
	'30_': 16, #problem
	'2_': 240, #problem
	'120_': 4,
	'20_': 24, #problem
	'240_': 2,
	'5_': 96,
	'6_': 80,
	'10_': 48, #problem
	'3_': 160,
	'60_': 8
}

list_with_indicators = [
	'SMA', 'SMM', 'EMA_13', 'EMA_26', 'EMA_DIF', 'DEMA', 'TEMA', 'TRIMA', 'TRIX',
	'VAMA', 'ER', 'ZLEMA', 'WMA', 'HMA', 'EVWMA', 'VWAP', 'SMMA', 'MOM',
	'ROC', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'PERCENT_B', 'ADX', 'STOCH', 
	'STOCHD', 'STOCHRSI', 'WILLIAMS', 'UO', 'AO', 'TP', 'ADL', 'CHAIKIN', 'MFI',
	'OBV', 'WOBV', 'VZO', 'EFI', 'CFI', 'EMV', 'CCI', 'COPP', 'CMO', 'FISH', 
	'SQZMI', 'VPT', 'FVE', 'VFI', 'MSD', 'return'
]

for asset in assets:
	
	file_with_model = asset + '_model.sav'
	best_clf = pickle.load(open(file_with_model, 'rb'))

	file_with_features = asset + '_features.sav'
	features = pickle.load(open(file_with_features, 'rb'))
	
	df = pd.read_sql(
		'''SELECT * FROM (SELECT * FROM {} ORDER BY date_time DESC LIMIT 300)Var1
		  ORDER BY date_time ASC'''.format(asset), engine
	)
	df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
	df = df.dropna()
	df = df.set_index('date_time')
	df = df.drop_duplicates()
	
	df['dist_to_max_per_range'] = np.array(df[['close']]\
		.iloc[::-1].rolling(30, min_periods=1).max().iloc[::-1])\
		- np.array(df[['close']])
	df['dist_to_min_per_range'] = np.array(df[['close']])\
		- np.array(df[['close']]\
		.iloc[::-1].rolling(30, min_periods=1).min().iloc[::-1])

	#Calculates proportion of each row in order book to the apropriate section(bid or offer)
	df_offer_count_proportion = df.loc[:, 'offer_count_10':'offer_count_1']\
		.div(df.loc[:, 'offer_count_10':'offer_count_1'].sum(axis=1), axis=0)
	df_bid_count_proportion = df.loc[:, 'bid_count_10':'bid_count_1']\
		.div(df.loc[:, 'bid_count_10':'bid_count_1'].sum(axis=1), axis=0)
	#Calculates offer/bid ratio per row
	offer_bid_ratio = pd.DataFrame(df.loc[:, 'offer_count_10':'offer_count_1'].sum(axis=1) /\
		df.loc[:, 'bid_count_10':'bid_count_1'].sum(axis=1))
	df = df.drop([
		'offer_count_10', 'offer_count_9', 'offer_count_8', 'offer_count_7',
		'offer_count_6', 'offer_count_5', 'offer_count_4', 'offer_count_3',
		'offer_count_2', 'offer_count_1', 'bid_count_10', 'bid_count_9', 
		'bid_count_8', 'bid_count_7',
		'bid_count_6', 'bid_count_5', 'bid_count_4', 'bid_count_3',
		'bid_count_2', 'bid_count_1'], axis = 1)

	#Concatenates single df for analysis
	list_of_dfs = [
		df,
		df_offer_count_proportion, 
		df_bid_count_proportion, 
		offer_bid_ratio
	]
	df_to_analyze = pd.concat(list_of_dfs, axis=1)

	df_to_analyze = df_to_analyze.dropna()
	
	print(asset)
	print(df)
	print()
	break
