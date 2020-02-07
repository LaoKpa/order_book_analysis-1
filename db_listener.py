'''
Data ingestion
'''
import pickle
from sqlalchemy import create_engine

'''
Statistical packages
'''
import pandas as pd
import numpy as np

'''
Custom functions
'''
from utils.append_indicators import append_indicators

'''
Creates MySQL connection object
'''
protocol_user_pass = 'mysql://Quotermain:Quotermain233@'
host_port_db = '192.168.0.105:3306/trading_data'
engine = create_engine(
    protocol_user_pass + host_port_db
)

'''
Creates collections with timeframes 
for candles and indicators
'''
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
    'SMA', 'SMM', 'EMA_13', 'EMA_26', 'EMA_DIF', 
	'DEMA', 'TEMA', 'TRIMA', 'TRIX', 'VAMA', 'ER', 
	'ZLEMA', 'WMA', 'HMA', 'EVWMA', 'VWAP', 'SMMA', 
	'MOM', 'ROC', 'RSI', 'IFT_RSI', 'TR', 'ATR', 
	'BBWIDTH', 'PERCENT_B', 'ADX', 'STOCH', 'STOCHD', 
	'STOCHRSI', 'WILLIAMS', 'UO', 'AO', 'TP', 'ADL', 
	'CHAIKIN', 'MFI', 'OBV', 'WOBV', 'VZO', 'EFI', 
	'CFI', 'EMV', 'CCI', 'COPP', 'CMO', 'FISH', 
    'SQZMI', 'VPT', 'FVE', 'VFI', 'MSD', 'return'
]

def run():
	
	'''
	Reads the LIMITED data for SBER
	'''
	query = '''
		SELECT * FROM (
			SELECT * FROM SBER_train 
			ORDER BY date_time DESC LIMIT 500
		)Var1
		ORDER BY date_time ASC
	'''
	df = pd.read_sql(query, engine)
	
	'''
	Sets the datetime index, drops
	duplicates and nulls
	'''
	df['date_time'] = pd.to_datetime(
		df['date_time'], errors='coerce'
	)
	df = df.set_index('date_time')
	df.dropna(inplace=True)
	
	'''
	Calculates proportion of each row 
	in order book to the apropriate 
	section(bid or offer)
	'''
	#Offer
	OC_cols = df.loc[
		:, 'offer_count_10':'offer_count_1'
	]
	df_offer_count_proportion =\
	  OC_cols.div(OC_cols.sum(axis=1), axis=0)
	#Bid
	BC_cols = df.loc[
		:, 'bid_count_10':'bid_count_1'
	]
	df_bid_count_proportion =\
      BC_cols.div(BC_cols.sum(axis=1), axis=0)
	
	'''
	Calculates offer/bid ratio per row
	'''
	offer_bid_ratio = pd.DataFrame(
		OC_cols.sum(axis=1) /\
		BC_cols.sum(axis=1))
	
	'''
	Drops columns with separate bids
	and asks
	'''
	cols_to_drop = [
		'offer_count_10', 'offer_count_9', 
		'offer_count_8', 'offer_count_7',
		'offer_count_6', 'offer_count_5', 
		'offer_count_4', 'offer_count_3',
		'offer_count_2', 'offer_count_1', 
		'bid_count_10', 'bid_count_9', 
		'bid_count_8', 'bid_count_7',
		'bid_count_6', 'bid_count_5', 
		'bid_count_4', 'bid_count_3',
		'bid_count_2', 'bid_count_1'
	]
	df.drop(cols_to_drop, axis=1, inplace=True)
	
	'''
	Concatenates single df for analysis
	and drops nulls
	'''
	list_of_dfs = [
		df,
		df_offer_count_proportion, 
		df_bid_count_proportion, 
		offer_bid_ratio
	]
	temp_df = pd.concat(list_of_dfs, axis=1)
	temp_df.dropna(inplace=True)

	'''
	Appends indicators and drops nulls
	'''
	for key in dict_of_tf:
		temp_df = append_indicators(
			temp_df, key, list_with_indicators
		)
	temp_df = temp_df.dropna()
	temp_df.shape
	
	print(temp_df.head(100))


if __name__ == '__main__':
	run()