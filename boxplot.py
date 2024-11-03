# import the required library 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# load the dataset 
df = pd.read_csv("tips.csv") 
df.boxplot(by ='day', column =['total_bill'], grid = False) 
print(df)
