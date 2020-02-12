'''
Data processing
'''
from joblib import load
from sqlalchemy import create_engine
import MySQLdb, datetime

'''
Statistical packages
'''
import pandas as pd
import numpy as np

'''
Custom functions
'''
from utils.append_indicators import append_indicators
from utils.predict import predict

'''
Constants
'''
PROBA_THRESH = 0.70
DIST_TO_MAX = 2
DIST_TO_MIN = 2
ASSET = 'SBER'

'''
Creates MySQL connection object
for reading the latest data from the DB
'''
protocol_user_pass = 'mysql://Quotermain:Quotermain233@'
host_port_db = '192.168.0.105:3306/trading_data'
engine = create_engine(
	protocol_user_pass + host_port_db
)

'''
Creates connection to the DB 
to write a signal
'''
db = MySQLdb.connect(
    host="192.168.0.105",
    port = 3306,
    user="Quotermain", 
    passwd="Quotermain233", 
    db="trading_data", 
    charset='utf8'
)
cursor = db.cursor()

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

'''
Uploads the model
'''
file_with_model = '/home/quotermin/ml/trading/' +\
    'candles_ticks_orderbook/'+ ASSET +'_model.joblib'
clf = load(file_with_model)

def run():
	
	'''
	Reads the LIMITED data for SBER
	'''
	query = '''
		SELECT * FROM (
			SELECT * FROM {}_train 
			ORDER BY date_time DESC LIMIT 3000
		)Var1
		ORDER BY date_time ASC
	'''.format(ASSET)
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
	
	print(temp_df.shape)
	
	'''
	Makes predictions from the latest uploaded data
	with shifted threshold
	'''
	y_pred = predict(clf, temp_df, PROBA_THRESH)
	
	'''
	Uploads a signal to the DB
	'''
	if y_pred[-1] == 'up':
		sql = """UPDATE `trade_signals`
			SET `signal` = 'long',
			`dist_to_max`={},
			`dist_to_min`={} 
			WHERE `asset`='{}'"""\
			.format(
				DIST_TO_MAX,
				DIST_TO_MIN,
				ASSET
			)
		cursor.execute(sql)
		db.commit()
		print(
			datetime.datetime.now().time(), 
			'Long ', 
			ASSET
		)
	elif y_pred[-1] == 'down':
		sql = """UPDATE `trade_signals`
			SET `signal` = 'short',
			`dist_to_max`={},
			`dist_to_min`={} 
			WHERE `asset`='{}'"""\
			.format(
				DIST_TO_MAX,
				DIST_TO_MIN,
				ASSET
			)
		cursor.execute(sql)
		db.commit()
		print(
			datetime.datetime.now().time(), 
			'Short ', 
			ASSET
		)
	elif y_pred[-1] == 'nothing':
		sql = """UPDATE `trade_signals` 
			SET `signal` = 'nothing',
			`dist_to_max`=0,
			`dist_to_min`=0
			WHERE `asset`='{}'""".format(ASSET)
		cursor.execute(sql)
		db.commit()
		print(
			datetime.datetime.now().time(), 
			'Nothing ', 
			ASSET
		)


if __name__ == '__main__':
	run()