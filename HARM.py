from aix360.algorithms.rbm import BooleanRuleCG
from aix360.algorithms.rbm import FeatureBinarizer
from ruleset import BayesianRuleSet
import pandas as pd
import numpy as np

class HARM_BRCG():

	def __init__(self, train_data, train_labels, black_box, feature_names = None):
		"""
		Uses BRCG as the interpretable classifier.
		black box must have sklearn fit and predict methods
		Assumes train_data is numpy array and needs to be binarized. 
		Assumes labels are int 0 or 1. 

		Reference for BRCG: Boolean Decision Rules via Column Generation - NeurIPS 2018
		"""
		self.black_box = black_box
		self.train_data = train_data
		self.train_labels = train_labels
		self.feature_names = feature_names

		self.r_0 = None
		self.r_1 = None
		self.binarized_train_data = None
		self.binarizer = None

	def binarize_data(self):
		"""
		This has to be run after initialization, to binarize the training data.
		Then, the binarizer is saved to binarize any points during 
		prediction time.
		"""
		fb = FeatureBinarizer(negations = True, returnOrd = True)
		dfTrain = pd.DataFrame(self.train_data) 
		if self.feature_names:
			dfTrain.columns = self.feature_names
		dfTrain, dfTrainStd = fb.fit_transform(dfTrain)
		self.binarizer = fb
		self.binarized_train_data = dfTrain
		return

	def train_black_box(self):
		"""
		Trains the black box.
		"""
		self.black_box.fit(self.train_data, self.train_labels)
		return

	def train_r_0(self):
		"""
		Trains the rule model that outputs rules that predict
		for class 0. 
		"""

		# the rules generated predict for label 0. 
		br_0 = BooleanRuleCG(CNF = False)
		br_0.fit(self.binarized_train_data, self.train_labels)
		self.r_0 = br_0
		return

	def train_r_1(self):
		"""
		Trains the rule model that outputs rules that predict
		for class 1.
		"""

		# the rules generated predict for label 0, so we hvae to 
		# invert the labels to generate rules that predict
		# for label 1.
		br_1 = BooleanRuleCG(CNF = False)
		inverted_train_labels = []
		for label in self.train_labels:
			if label:
				inverted_train_labels.append(0)
			else:
				inverted_train_labels.append(1)

		br_1.fit(self.binarized_train_data, np.array(inverted_train_labels))
		self.r_1 = br_1
		return

	def r_0_predict(self, test_data_df):
		"""
		Called in predict method. Returns prediction of r0.
		"""
		return self.r_0.predict(test_data_df)

	def r_1_predict(self, test_data_df):
		"""
		Called in predict method. Returns prediction of r1.
		"""
		inverse_predictions = self.r_1.predict(test_data_df)
		
		# the rules we got are to predict to true class 0 if they hold. 
		# Because BRCG is implemented this way, we trained it on the
		# inverted labels to get rules that would predict class 1 if they hold.
		# So now, when we want to get the predictions of r1, we need to flip
		# the predictions of the model.

		r_1_predictions = []
		for p in inverse_predictions:
		 	if not p:
		 		r_1_predictions.append(1)
		 	else:
		 		r_1_predictions.append(0)
		return r_1_predictions

	def predict(self, test_data):
		"""
		Takes in a numpy array of test_data. Binarizes it.
		Returns the predictions and the ratio of predictions that
		did not require the black box over the ratio of predictions
		that required calling the black box.
		""" 
		dfTest = pd.DataFrame(test_data)
		if self.feature_names:
			dfTest.columns = self.feature_names

		binarized_test_data, _ = self.binarizer.transform(dfTest)

		# Get the rule models to predict for every point first.
		r_0_pred = self.r_0_predict(binarized_test_data)
		r_1_pred = self.r_1_predict(binarized_test_data)

		# Once both rule based models have a prediction, see where
		# they disagree then call the black box there.
		final_predictions = []
		transparent_calls = 0
		total_calls = 0
		for i, pred in enumerate(r_0_pred):
			if r_1_pred[i] == pred:
				final_predictions.append(pred)
				transparent_calls += 1
			else:
				final_predictions.append(self.black_box.predict([test_data[i]])[0])

			total_calls += 1

		return transparent_calls, total_calls, final_predictions
