# import library:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import sales data
s = pd.read_csv('tips.csv')

# defining the outlier function for interquartile range
def outlier_zscore(df_column):
    global outlier, z_score
    outlier = []
    z_score = []
    threshold = 3
    mean = np.mean(df_column)
    std = np.std(df_column)
    for i in df_column:
        zscore = (i - mean)/std
        z_score.append(zscore)
        if np.abs(zscore) > threshold:
            outlier.append(i)
    return print("total number of outliers", len(outlier))
# getting total number of outlier
outlier_zscore(s.tip)
# plotting figure
plt.figure(figsize = (10,5))
sns.distplot(z_score)
# creting band to identify the outliers
# plt.axvspan : function sets the vertical rectangle across the axes of the plot
plt.axvspan(xmin = 3 ,xmax = max(z_score),alpha = 0.25, color ='red')
