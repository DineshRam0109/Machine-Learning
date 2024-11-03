import pandas as pd
import numpy as np
# Function to calculate prior probabilities and likelihoods
def train_naive_bayes(df, feature_cols, target_col):
prior_prob=df[target_col].value_counts(normalize=True).to_dict()
likelihood={}
for cls in prior_prob.keys():
cls_data=df[df[target_col]==cls]
likelihood[cls]={}
for feature in feature_cols:
likelihood[cls][feature] = {
'mean':cls_data[feature].mean(),
'std':cls_data[feature].std()
}
return prior_prob,likelihood
# Function to calculate Gaussian probability density function
def gaussian_prob(x,mean,std):
exponent=np.exp(-((x-mean)**2)/(2*(std**2)))
return(1 /(np.sqrt(2*np.pi)*std))*exponent
# Function to predict the class of a new instance
def naive_bayes_predict(instance, prior_prob, likelihood, feature_cols):
post_prob={}
for cls in prior_prob.keys():
post_prob[cls]=prior_prob[cls]
for feature, value in zip(feature_cols, instance):
mean=likelihood[cls][feature]['mean']
std=likelihood[cls][feature]['std']
post_prob[cls]*=gaussian_prob(value,mean,std)
predicted_class=max(post_prob, key=post_prob.get)
return predicted_class
# Train the model
feature_cols=['age','cp','trestbps','chol','thalach','oldpeak']
target_col='target'
prior_prob,likelihood=train_naive_bayes(df,feature_cols,target_col)
# New instance
new_instance=[50, 1, 120, 240, 150, 2.3]
#new_instance= [40, 0, 110, 200, 160, 1.0]
predicted_class = naive_bayes_predict(new_instance,prior_prob,likelihood,feature_cols)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("Prediction for new instance:", "Yes, There is Heart Disease" if predicted_class == 1 else "No Heart Disease")