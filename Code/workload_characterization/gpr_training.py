
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from gpr_model import GPRNP
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#load workload file names
def loadWorkloadFileNames(folder):
	onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
	return onlyfiles

def loadWorkload(file_name):
	if ".xlsx" in file_name:
		return None
	df = pd.read_pickle(file_name)
	return df
#split the columns individually into X train data and Y train predictions.
def splitColData(df):
	cols_list = list(df.columns)
	X_train = df[cols_list[1:13]].values
	Y_train = df[cols_list[13:]].values
	return X_train,Y_train
	

#training all the different GPR models required (no of workloads * no of metrics) models
def trainWorkloads():
	pruned_file ="../../Data/PrunedWorkloadFiles/"
	#find all the files in workload
	files = loadWorkloadFileNames(pruned_file)
	X_whole,Y_whole = splitColData(loadWorkload("../../Data/ConcatPrunedFile.pkl"))
	 
 
	X_scaler = StandardScaler()
	X_scaler.fit(X_whole)

	y_scaler = StandardScaler(copy=False)
	y_scaler.fit_transform(Y_whole)

	for wl_file in files:
		wrk_load = loadWorkload(pruned_file+wl_file)
		print("File name:",wl_file)
		if wrk_load is None:
			continue
		X_train,Y_train = splitColData(wrk_load)
		print("X train:",X_train.shape," Y train:",Y_train.shape)
		
		X_scaled = X_train #X_scaler.transform(X_train)

		for metrics in range(0,len(Y_train[0])):

			


			y_scaled = Y_train #y_scaler.transform(Y_train)


			Y_train_metric = y_scaled[:,metrics:metrics+1]
			print("inputting to model ",Y_train_metric.shape)
			model = GPRNP()
			for epoch in range(0,1):
				model.fit(X_scaled, Y_train_metric, ridge=1.0)
				gpr_result = model.predict(X_scaled)
				predictions = gpr_result.ypreds.ravel()
				#print("preds:",predictions)
				#print("True:",Y_train_metric)
				print("Error:",np.sum((predictions-Y_train_metric)**2))

trainWorkloads()
