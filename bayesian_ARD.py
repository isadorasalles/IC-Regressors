import pandas as pd 
import numpy as np
import os
import glob 
import time

for folder in glob.glob("/home/isadorasalles/Documents/Regressao/train_test/*"):  #para percorrer por todas as pastas da pasta

	os.chdir(folder)
	name_folder = folder.split("/")[6]
	train_data = np.array(pd.read_csv('train_data.csv', sep= ';'))
	test_data = np.array(pd.read_csv('test_data.csv', sep= ';'))
	train_labels = np.array(pd.read_csv('train_labels.csv', sep= ';'))
	test_labels = np.array(pd.read_csv('test_labels.csv', sep= ';'))

	inicio = time.time()

	from sklearn.linear_model import ARDRegression 

	# treinar o modelo no conjunto de dados
	regression = ARDRegression().fit(train_data, train_labels)

	# prever 
	predictions_labels = regression.predict(test_data)

	fim = time.time()
	df_time = pd.DataFrame({'Execution Time:' : [fim-inicio]})

	output_path = os.path.join('/home/isadorasalles/Documents/Regressao/bayesian_ARD', 'time_'+name_folder)
	df_time.to_csv(output_path, sep=';')

	from sklearn import metrics

	df_metrics = pd.DataFrame({'Mean Absolute Error' : [metrics.mean_absolute_error(test_labels, predictions_labels)], 'Mean Squared Error' : [metrics.mean_squared_error(test_labels, predictions_labels)],  
		'Root Mean Squared Error': [np.sqrt(metrics.mean_squared_error(test_labels, predictions_labels))], 'R2 Score': [metrics.r2_score(test_labels, predictions_labels)]})

	output_path = os.path.join('/home/isadorasalles/Documents/Regressao/bayesian_ARD', 'metrics_'+name_folder)
	df_metrics.to_csv(output_path, sep=';')
