import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 # import sales data
s = pd.read_csv('tips.csv')

# defining the outlier function for interquartile range
def outlier_IQR(df, column):
    global lower, upper
    Q1 = np.quantile(df[column], 0.25) # first quartile
    Q3 = np.quantile(df[column], 0.75) # third quartile
    IQR = Q3 - Q1 # inter - quartile range
    threshold = 1.5 * IQR # defining the threshold
    lower = Q1 - threshold 
    upper = Q3 + threshold
    lower_bound = df[df[column] < lower]
    upper_bound = df[df[column] > upper]
    #printing IQR, threshold, lower bound, upper bound and total number of outlier
    print('IQR is:', IQR)
    print('Threshold is:', threshold)
    print('Lower bound is:', lower)
    print('Upper bound is:', upper)
    return print('total number of outliers are:', lower_bound.shape[0] + upper_bound.shape[0])
# getting the IQR, threshold, lower bound, upper bound, total outliers
outlier_IQR(s, 'tip')
# plotting figure
plt.figure(figsize = (10,8))
sns.distplot(s.tip, bins = 25)
# creting band to identify the outliers
# plt.axvspan : function sets the vertical rectangle across the axes of the plot
plt.axvspan(xmin = lower, xmax = s.tip.min(), alpha = 0.2, color = 'red')
plt.axvspan(xmin = upper, xmax = s.tip.max(), alpha = 0.2, color = 'red')

