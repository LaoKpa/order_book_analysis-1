import pickle, os, glob, csv, datetime, re
import pandas as pd
import numpy as np
from finta import TA
from sklearn.metrics import classification_report
from utils.append_indicators import append_indicators


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
    #'RTKM', 
    'SBER', 
    'SBERP', 
    'SIBN', 
    #'SNGS', 
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


while True:
			
	for asset in assets:
		
		try:

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
					dict_with_tf['1_'],
					dict_with_tf['4_'],
					on='date_time'
				)
				del(dict_with_tf['1_'])
				del(dict_with_tf['4_'])
			except:
				print(asset)
				print(key)
				pass
				#continue

			for key in dict_with_tf:
				if key != '1_' or key != '4_':
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
			df_to_analyze = df_to_analyze[features]

			df_to_analyze['Return'] = df_to_analyze['1_return'].shift(-1)
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

			up_prec = classification_report(
				y_test, y_pred, output_dict=True
			)['up']['precision']
			down_prec = classification_report(
				y_test, y_pred, output_dict=True
			)['down']['precision']
			nothing_prec = classification_report(
				y_test, y_pred, output_dict=True
			)['nothing']['precision']


			if y_pred[-1] == 'up':
				with open('test/' + asset + '/signal.csv', mode='w') as signal_file:
					signal_writer = csv.writer(
						signal_file, 
						delimiter=','
					)
					signal_writer.writerow(
						[
							'long'
						]
					)
				print(datetime.datetime.now().time(), 'Long ', asset)
			elif y_pred[-1] == 'down':
				with open('test/' + asset + '/signal.csv', mode='w') as signal_file:
					signal_writer = csv.writer(
						signal_file, 
						delimiter=','
					)
					signal_writer.writerow(
						[
							'short'
						]
					)
				print(datetime.datetime.now().time(), 'Short ', asset)
			elif y_pred[-1] == 'nothing':
				with open('test/' + asset + '/signal.csv', mode='w') as signal_file:
					signal_writer = csv.writer(
						signal_file, 
						delimiter=','
					)
					signal_writer.writerow(
						[
							'nothing'
						]
					)
				print(datetime.datetime.now().time(), 'Nothing ', asset)


			print(asset)
			print(classification_report(y_test, y_pred))
			print()
		
		except (ValueError, EOFError, csv.Error, KeyError, IndexError) as e:
			print(datetime.datetime.now().time(), 'Lets try again')
			continue
		#break



