import pickle, os, glob, csv, datetime, re
import pandas as pd
import numpy as np
from finta import TA
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from time import sleep

os.chdir('/home/quotermin/Desktop/Windows-Share/')

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
	'4_': 120,
	'15_': 32,
	'30_': 16, #problem
	'2_': 240, #problem
	'120_': 4,
	'20_': 24, #problem
	'240_': 2,
	'5_': 96,
	#'1_': 480, #problem
	'6_': 80,
	'10_': 48, #problem
	'3_': 160,
	'60_': 8
}


def append_indicators(df, tf, indicators):
	ohlcv = df.loc[:,[
		tf + 'open', tf + 'high', tf + 'low', 'close', tf + 'volume'
	]]
	ohlcv = ohlcv.rename(columns = {
		tf + 'open': 'open',
		tf + 'high': 'high',
		tf + 'low': 'low',
		tf + 'volume': 'volume'
	})
	for indicator in indicators:
		if indicator == 'EMA_DIF':
			EMA_13 = eval(
				'TA.EMA(ohlcv, period = 13)'
			)
			EMA_26 = eval(
				'TA.EMA(ohlcv, period = 26)'
			)
			df[tf + indicator] = EMA_13 - EMA_26
		elif indicator == 'EMA_13':
			df[tf + indicator] = eval(
				'TA.EMA(ohlcv, period = 13)'
			)
		elif indicator == 'EMA_26':
			df[tf + indicator] = eval(
				'TA.EMA(ohlcv, period = 26)'
			)
		elif indicator == 'PERCENT':
			df[tf + indicator + '_B'] = eval(
				'TA.PERCENT_B(ohlcv)'
			)
		elif indicator == 'IFT':
			df[tf + indicator + '_RSI'] = eval(
				'TA.IFT_RSI(ohlcv)'
			)
		elif indicator == 'return':
			df[tf + 'return'] = ohlcv['close'].pct_change(1)
		else:
			df[tf + indicator] = eval('TA.' + indicator + '(ohlcv)')

	return df
	
	
def run():
	
	for asset in assets:

		file_with_model = asset + '_model.sav'
		best_clf = pickle.load(open(file_with_model, 'rb'))
		
		file_with_features = asset + '_features.sav'
		features = pickle.load(open(file_with_features, 'rb'))
		
		list_with_tf = list()
		for feature_name in features:
			tf = re.search('^[0-9]*_', feature_name)
			if tf:
				if tf.group(0) not in list_with_tf:
					list_with_tf.append(tf.group(0))
		
		dict_with_tf = dict()
		
		for tf in list_with_tf:
			file_with_data = glob.glob('test/' + asset + '/' + tf + '*')[0]
			df_right = pd.read_csv(
				file_with_data,
				names = ['date_time', 'high', 'low', 'open', 'close', 'volume']
			)
			df_right['date_time'] = df_right['date_time']\
				.replace(r'^16.*', np.NaN, regex=True)
			df_right = df_right.dropna()
			df_right['date_time'] = pd.to_datetime(df_right['date_time'])
			
			list_with_indicators = list()
			for feature_name in features:
				indicator = re.search(
					tf + '([a-zA-Z]+_?\w*)', 
					feature_name
				)
				if indicator:
					if indicator not in list_with_indicators:
						list_with_indicators.append(indicator.group(1))
			
			df_right = append_indicators(
				df_right, tf, list_with_indicators
			)
			df_right = df_right.drop([
				'high', 'low', 'open', 'volume', 'close'
			], axis=1)
			dict_with_tf[tf] = df_right
			
			#print(asset)
			#print(tf)
		
		try:
			df_to_analyze = pd.merge_asof(
				dict_with_tf['15_'],
				dict_with_tf['4_'],
				on='date_time'
			)
			del(dict_with_tf['15_'])
			del(dict_with_tf['4_'])
		except:
			print(asset)
			print(key)
			pass
			#continue

		for key in dict_with_tf:
			if key != '15_' or key != '4_':
				try:
					df_to_analyze = pd.merge_asof(
						df_to_analyze,
						dict_with_tf[key],
						on='date_time'
					)
				except:
					print(asset)
					print(key)
					pass

		new_features = set(df_to_analyze.columns) & set(features)
		new_features = list(new_features)
		df_to_analyze = df_to_analyze[new_features]
		
		df_to_analyze['Return'] = df_to_analyze['15_return'].shift(-1)
		conditions = [
			df_to_analyze['Return'] > 0,
			df_to_analyze['Return'] < 0
		]
		choices = ['up', 'down']
		df_to_analyze['y'] = np.select(conditions, choices, default='nothing')
		df_to_analyze = df_to_analyze.dropna()

		X_test = df_to_analyze.drop(['y', 'Return'], axis=1)
		y_test = df_to_analyze['y']
		
		y_pred = best_clf.predict(X_test)
		
		print(asset)
		print(classification_report(y_test, y_pred))
		print()
		
		#break

		
		

if __name__ == '__main__':
	run()