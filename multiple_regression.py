#  Peter Stephens
#  5/28/2016

# Multivariate Analysis
#
# Use Lending Club Statistics for (annual_inc) to model interest rates (int_rate).
# Also, add home ownership (home_ownership) to the model.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

#  Read in Lending Club Data form git hub repository
loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')

#  Clean Data:  Remove null value rows
loansData.dropna(inplace=True)

ammount_requested = loansData['Amount.Requested'].map(lambda x: float(x))
interest_rate =  loansData['Interest.Rate'].map(lambda x: float(x.rstrip('%')))
loansData['Annual.Income']    = loansData['Monthly.Income'].map(lambda x: float(x * 12.0))
annual_income = loansData['Annual.Income'] 


# The dependent variable
y = np.matrix(interest_rate).transpose()

################################################################
#  Use income (annual_inc) to model interest rates (int_rate). #
#  Don't forget (ammount_requested).                           #
################################################################

# The independent variables shaped as columns
x1 = np.matrix(annual_income).transpose()
x2 = np.matrix(ammount_requested).transpose()
x = np.column_stack([x1, x2])

#  Create a linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

#  Output the Regression Summary
print(f.summary())

################################################################
#  Add home ownership (home_ownership) to the model.           #
################################################################

home_ownership = loansData['Home.Ownership'].map(lambda x: (1.0 if x == 'MORTGAGE' else 0.0))

# The independent variables shaped as columns
x1 = np.matrix(annual_income).transpose()
x2 = np.matrix(ammount_requested).transpose()
x3 = np.matrix(home_ownership).transpose()
x = np.column_stack([x1, x2, x3])

#  Create a linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

#  Output the Regression Summary
print(f.summary())



