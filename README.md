# PuppyName
A warehouse based on Baidu dog competition.
# Notes #
This is just an example of a simple transferring learning. There are other, but not yet updated.

In the Baidu dog game, this simple transferring learning method. Only achieved a correct rate of 0.78.
# Aequirements#
	 1.Python 3.5/Python 2.7
	 2.TensorFlow
	 3.Pillow, Jupyter Notebook etc.
	 4.OS
	   Windows or Ubuntu

	
# Steps #

	 1.New folder
		$ mkdir train
		$ mkdir val
		The training set and validation set into the corresponding folder.
	 2.Preprocess data
		$ python preprocess_data_v2.py
	 3.Training
		$ python inception.py
	 4.Evaluation
		$ python eval.py
	 5.Notes
		Before you can execute the program, you have to change the file path accordingly.
