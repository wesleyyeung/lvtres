# LVTRES: Prediction of LV Thrombus Resolution

Welcome to the project repo for the LVTRES (Left Ventricular Thrombus Resolution). The aim of this project is to develop a prediction model using data from patients who develop post-acute myocardial infarction (AMI) left ventricular thrombus to predict resolution of thrombus versus a composite outcome of death, recurrence of thrombus and persistent thrombus on serial echocardiography. 

# Getting Started

This project was developed in Python 3.7. A full list of packages used are in the 'requirements.txt' file.

# Navigation

There are 3 Jupyter notebooks that are meant to be run in the following order:
1. descriptive.ipynb
2. train.ipynb
3. evaluate.ipynb

The first notebook, along with the accompanying 'utils.py' contains code used for preprocessing the data and calculating descriptive statistics. The second notebook contains code used to develop the model. The third notebook contains code used to evaluate the model on the held out test set. 

A pickled file containing the final Sci-kit learn model is in the 'pickled_objects' directory

# License

This project is licensed under the MIT License - see the LICENSE.md file for details
