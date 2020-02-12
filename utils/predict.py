import numpy as np

def predict(model, features, thresh):
	
	
	# Predicts probabilities
	y_pred_proba = model.predict_proba(features)

	# Creates an empty 1D string array 
	# and fills it with default string
	y_pred = np.empty(
		len(y_pred_proba), dtype=object, order='C'
	)
	y_pred[:] = 'nothing'

	# Fills the array with predictions 
	# according threshold
	y_pred[
		np.where(y_pred_proba[:, 0] >= thresh)
	] = 'down'
	y_pred[
		np.where(y_pred_proba[:, 2] >= thresh)
	] = 'up'

	return y_pred
