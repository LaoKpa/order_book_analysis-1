#Data ingestion
import pickle
from sqlalchemy import create_engine

#Statistical packages
import pandas as pd
import numpy as np

#Custom functions
from utils.append_indicators import append_indicators

#Creates MySQL connection object
protocol_user_pass = 'mysql://Quotermain:Quotermain233@'
host_port_db = '192.168.0.105:3306/trading_data'
engine = create_engine(
    protocol_user_pass + host_port_db
)

def run():
	
	#Reads the LIMITED data for SBER
	query = 'SELECT * FROM SBER_train LIMIT 500'
	df = pd.read_sql(query, engine)
	
	print(df.head())


if __name__ == '__main__':
	run()