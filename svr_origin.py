'''
Usage: svr.py filename [options]

Options:
	--version             					show program's version number and exit
	-h, --help           				 	show this help message and exit
	-l LOOK_BACK, --lookback=LOOK_BACK		sliding window size
	-p TEST_SIZE, --trsize=TEST_SIZE		training set size
'''

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

import sys
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV
import time
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from optparse import OptionParser
import math
from time import gmtime, strftime
import os
from a2f import arrays2file


ttime = strftime("%Y%m%d-%H%M%S", gmtime())
sys.stdout = open("./log/svr_origin_%s.log" %ttime, "w")

def manage_options():
	parser = OptionParser(usage="usage: %prog [options] filename", version="%prog 1.0")
	parser.add_option("-l", "--lookback",
						dest="look_back",
						type="int",
						default=30,
						help="sliding window size")
	parser.add_option("-p", "--tesize",
						dest="test_size",
						type="int",
						default=7,
						help="test set size")
	parser.add_option("-n", "--nofuture",
						dest="future",
						action="store_true",
						default=False,
						help="no 'one step prediction'")
	(options, args) = parser.parse_args()
	if len(args) == 0:
		parser.error("wrong arguments")
	
	out_dict = vars(options)
	out_dict["file_train"] = args[0]
	return out_dict
	

#INPUT	
options = manage_options()
print "Options:", options

file_train = options["file_train"]
look_back = options["look_back"]
test_size = options["test_size"]
future = options["future"]

#LOAD
y = np.load(file_train)
x_init = range(y.shape[0])
sequence = y
sequence = sequence.reshape(sequence.shape[0], 1)

#NORMALIZE
scaler = MinMaxScaler(feature_range=(0, 1))
sequence = scaler.fit_transform(sequence)

#TRAIN-TEST SPLIT
train_size = len(sequence) - test_size
print "Training set size :", train_size 
print "Test set size :", test_size

# convert an array of values into a dataset matrix
def create_dataset(sequence, train_size, look_back):
	X_tr, y_tr, t = [], [], []
	tr = sequence[0:train_size, :]
	for i in range(len(tr) - look_back - 1):
		a = tr[i:(i+look_back), 0]
		if 0 not in a:
			X_tr.append(a)
			y_tr.append(tr[i + look_back, 0])
			t.append(i+look_back)
	
	X_te, y_te = [], []
	te = sequence[train_size-look_back:, :]
	for i in range(len(te) - look_back):
		a = te[i:(i+look_back), 0]
		if 0 not in a:
			X_te.append(a)
			y_te.append(te[i + look_back, 0])
		
	return (np.array(X_tr), np.array(y_tr)), (np.array(X_te), np.array(y_te)), t


# reshape into X=t and Y=t+1
(X_tr, y_tr), (X_te, y_te), ttr = create_dataset(sequence, train_size, look_back)


grid = {"C": [10**e for e in range(-1, 6)], "gamma": [10**e for e in range(-5,-1)], "epsilon": [10**e for e in range(-3,-1)]}
#grid = {"C": [1000], "gamma": [0.01], "epsilon": [0.01]}
print "Grid", grid

#svr = GridSearchCV(SVR(kernel='linear'), cv=5, param_grid={"C": [1e-1, 1e0, 1e1, 1e2, 1e3]}, verbose=True, n_jobs=8)
svr = GridSearchCV(SVR(kernel='rbf'), cv=5, param_grid=grid, verbose=False, n_jobs=8)
#svr = GridSearchCV(SVR(kernel='poly'), cv=5, param_grid={"C": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "degree": [2,3]}, verbose=True, n_jobs=8)

t0 = time.time()
svr.fit(X_tr, y_tr)
svr_fit = time.time() - t0

print "SVR model fitted in %.3f s"% svr_fit

print "Best params:", svr.best_params_
print "Best score:", svr.best_score_

# PREDICTION
y_tr_pred = svr.predict(X_tr)
print y_tr_pred.shape
if not future:
	y_te_pred = svr.predict(X_te)
else:
	x = X_te[0:1,:]
	y_te_pred = []
	for i in range(X_te.shape[0]):
		pred = svr.predict(x)
		y_te_pred.append(pred[0])
		x = np.append(x[0:1,1:],pred).reshape((1,x.shape[1]))
	y_te_pred = np.array(y_te_pred)
	print y_te_pred.shape	
	
# RESCALE BACK
y_tr_pred = scaler.inverse_transform([y_tr_pred]).flatten()
y_tr = scaler.inverse_transform([y_tr]).flatten()
y_te_pred = scaler.inverse_transform([y_te_pred]).flatten()
y_te = scaler.inverse_transform([y_te]).flatten()

# RMSE
tr_score = math.sqrt(mean_squared_error(y_tr, y_tr_pred))
print('Train score: %.2f RMSE' % (tr_score))
te_score = math.sqrt(mean_squared_error(y_te, y_te_pred))
print('Test score: %.2f RMSE' % (te_score))
print('Test score: %.2f RRMSE' % (te_score / y_te.sum() * 100.))
print('Test score: %.2f RRMSE' % (te_score * math.sqrt(y_te.shape[0]) / y_te.sum() * 100.))


# shift train predictions for plotting
y_tr_pred_plot = np.empty_like(sequence).flatten()
y_tr_pred_plot[:] = np.nan
y_tr_pred_plot[ttr] = y_tr_pred
# shift test predictions for plotting
y_te_pred_plot = np.empty_like(sequence).flatten()
y_te_pred_plot[:] = np.nan
y_te_pred_plot[-test_size:] = y_te_pred
# plot baseline and predictions
seq_plot = sequence.copy()
seq_plot = scaler.inverse_transform(seq_plot)
seq_plot[sequence==0] = 'nan'
plt.plot(seq_plot, label='actual data')
plt.plot(x_init, y_tr_pred_plot, label='training pred')
plt.plot(x_init, y_te_pred_plot, label='test pred')
plt.axvline(x=train_size-.5, color='k', linestyle='--', linewidth=1)
plt.xlabel('time')
plt.ylabel('value')
plt.title('SVR (RMSE:%.2f)' %te_score)
plt.legend()
plt.savefig('out/svr_orig_full_plot_%s.eps' %ttime, format='eps', dpi=300)
plt.show()
plt.clf()


head = 3
y_tr_pred_plot = y_tr_pred_plot[-head-test_size:]
y_te_pred_plot = y_te_pred_plot[-head-test_size:]
y_real_plot = scaler.inverse_transform(sequence)[-head-test_size:]
plt.axvline(x=head-.5, color='k', linestyle='--', linewidth=1)

plt.plot(y_real_plot, label='actual data')
#plt.plot(y_tr_pred_plot, label='training pred')
plt.plot(y_te_pred_plot, label='test pred')
plt.xlabel('time')
plt.ylabel('value')
plt.title('SVR')
plt.legend()
plt.savefig('out/svr_orig_zoom_plot_%s.eps' %ttime, format='eps', dpi=300)
plt.show()

arrays2file([np.array(range(y_real_plot.shape[0])), y_real_plot[:,0], y_te_pred_plot], "out/svr_orig_zoom_plot_%s.txt" %ttime)

sys.stdout.flush()
