#!/usr/bin/env python
# coding: utf-8

# # 0 Environment Setup

# In[1]:


# Change output format
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Set work directory
import os
os.getcwd()
os.chdir('/Users/bryan/Documents/Python/Data')
os.getcwd()

# Import packages
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# # 1 Data Exploration

# ## 1.1 Load data and preview

# In[2]:


raw_df = pd.read_csv('bank.csv')
raw_df.head() # dataset preview
print ("Num of rows: " + str(raw_df.shape[0])) # row count
print ("Num of columns: " + str(raw_df.shape[1])) # col count
#raw_df.info()
#raw_df.describe()


# ## 1.2 Data cleaning

# In[3]:


# Categorical feature checking
df = raw_df
df['Gender'][0]

# Whitespaces removal
df['Surname'] = df['Surname'].map(lambda x: x.strip())
df['Geography'] = df['Geography'].map(lambda x: x.strip())
df['Gender'] = df['Gender'].map(lambda x: x.strip())


# In[4]:


# Missing values 

# Checking
df.isnull().sum() # NA in features
#df.isnull().any(axis=1) # NA in rows

# Dropping
#df.dropna(axis='index') # drop the rows with NA
#df.dropna(thresh=5) # drop the rows with at least 5 values

# Filling
#df.fillna(0) # fill with 0
#df["preMLScore"].fillna(df["preMLScore"].median(), inplace=True) # fill with median
#df["postMLScore"].fillna(df.groupby("gender")["postMLScore"].transform("mean"), inplace=True) # fill with median by group


# In[5]:


# Duplication

# Checking
df.duplicated().sum()
#np.where(df.duplicated()) # Positions of duplications

# Dropping
#df=df.drop_duplicates()


# In[6]:


# Outlier
#sns.boxplot(df['Balance'],orient='v')


# # 2 Feature Engineering

# ## 2.1 Feature exploration 

# In[7]:


# Check the distribution of features
sns.distplot(df['CreditScore'])


# In[8]:


# correlations between all the features
corr = df[["RowNumber", "CustomerId", "Surname",
                    "CreditScore", "Geography", "Gender",
                    "Age", "Tenure", "Balance",
                    "NumOfProducts", "HasCrCard", "IsActiveMember",
                    "EstimatedSalary", "Exited"]].corr()

# show heapmap of correlations
sns.heatmap(corr)


# In[9]:


# check the actual values of correlations
corr


# In[10]:


# calculate two features correlation
from scipy.stats import pearsonr
print (pearsonr(df['Age'], df['Exited'])[0])


# ## 2.2 Feature preprocessiing

# In[11]:


df.head()


# In[12]:


# One-hot encoding
df = pd.get_dummies(df, columns=['Geography','Gender'])
df.head()
#test_df = df
#pd.get_dummies(test_df, columns=['Gender'])
#pd.get_dummies(test_df, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)


# In[13]:


# Get ground truth data

#y = np.where(df['Exited'] == 'True.',1,0)
y = df.Exited
y

# Drop some useless columns
to_drop = ['RowNumber','CustomerId','Surname','Exited']
feat_space = df.drop(to_drop, axis=1)

# Convert some features to boolean values
#yes_no_cols = ["intl_plan","voice_mail_plan"]
#churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

X = feat_space
X.head()


# In[14]:


# Check the propotion of y = 1
print(y.sum() / y.shape * 100)


# In[15]:


# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X


# # 3 Model Training

# ## 3.1 Split dataset

# In[16]:


# Splite data into training and testing
from sklearn import model_selection

# Reserve 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

print('training data has %d observations with %d features'% X_train.shape)
print('test data has %d observations with %d features'% X_test.shape)


# ## 3.2 Model training and selection

# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression

# Logistic Regression
classifier_logistic = LogisticRegression()

# K Nearest Neighbors
classifier_KNN = KNeighborsClassifier()

# Random Forest
classifier_RF = RandomForestClassifier()


# #### Logistic regression

# In[18]:


# Train the model - logistic
classifier_logistic.fit(X_train, y_train)

# Prediction of test data
classifier_logistic.predict(X_test)

# Accuracy of test data
classifier_logistic.score(X_test, y_test)


# #### KNN

# In[19]:


# Train the model - KNN
classifier_KNN.fit(X_train, y_train)

# Prediction of test data
classifier_KNN.predict(X_test)

# Accuracy of test data
classifier_KNN.score(X_test, y_test)


# #### Random Forest

# In[20]:


# Train the model - Random forest
classifier_RF.fit(X_train, y_train)

# Prediction of test data
classifier_RF.predict(X_test)

# Accuracy of test data
classifier_RF.score(X_test, y_test)


# In[21]:


# Use 5-fold Cross Validation to get the accuracy for the 3 models
model_names = ['Logistic Regression','KNN','Random Forest']
model_list = [classifier_logistic, classifier_KNN, classifier_RF]
count = 0

for classifier in model_list:
    cv_score = model_selection.cross_val_score(classifier, X_train, y_train, cv=5)
    # print(cv_score)
    print('Model accuracy of %s is: %.3f'%(model_names[count],cv_score.mean()))
    count += 1


# # 4 Model Evaluation

# ## 4.1 Confusion matrix

# In[22]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Helper functions

# calculate accuracy, precision and recall
def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print (classifier)
    print ("Accuracy is: %0.3f" % accuracy)
    print ("precision is: %0.3f" % precision)
    print ("recall is: %0.3f" % recall)

# print out confusion matrices
def draw_confusion_matrices(confusion_matricies):
    class_names = ['Not','Churn']
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        cal_evaluation(classifier, cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, interpolation='nearest',cmap=plt.get_cmap('Reds'))
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


# In[23]:


# Confusion matrix, accuracy, precison and recall for random forest and logistic regression
confusion_matrices = [
    ("Random Forest", confusion_matrix(y_test,classifier_RF.predict(X_test))),
    ("Logistic Regression", confusion_matrix(y_test,classifier_logistic.predict(X_test))),
]

draw_confusion_matrices(confusion_matrices)


# ## 4.2 ROC & AUC

# #### Random Forest

# In[24]:


from sklearn.metrics import roc_curve
from sklearn import metrics

# Use predict_proba to get the probability results of Random Forest
y_pred_rf = classifier_RF.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)


# In[25]:


# ROC curve of Random Forest result
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - RF model')
plt.legend(loc='best')
plt.show()


# In[26]:


from sklearn import metrics

# AUC score
metrics.auc(fpr_rf,tpr_rf)


# #### Logistic Regression

# In[27]:


# Use predict_proba to get the probability results of Logistic Regression
y_pred_lr = classifier_logistic.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)


# In[28]:


# ROC Curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - LR Model')
plt.legend(loc='best')
plt.show()


# In[29]:


# AUC score
metrics.auc(fpr_lr,tpr_lr)


# # 5 Feature Selection

# #### logistic regression

# In[30]:


# add L2 regularization to logistic regression
# check the coef for feature selection
scaler = StandardScaler()
X_l2 = scaler.fit_transform(X)
LRmodel_l2 = LogisticRegression(penalty="l2", C = 5)
LRmodel_l2.fit(X_l2, y)
LRmodel_l2.coef_[0]
print ("Logistic Regression (L2) Coefficients")
for k,v in sorted(zip(map(lambda x: round(x, 4), LRmodel_l2.coef_[0]),                       feat_space.columns), key=lambda k_v:(-abs(k_v[0]),k_v[1])):
    print (v + ": " + str(k))


# #### random forest

# In[31]:


# check feature importance of random forest for feature selection
forest = RandomForestClassifier()
forest.fit(X, y)

importances = forest.feature_importances_

# Print the feature ranking
print("Feature importance ranking by Random Forest Model:")
for k,v in sorted(zip(map(lambda x: round(x, 4), importances), feat_space.columns), reverse=True):
    print (v + ": " + str(k))


# In[32]:


sns.pairplot(df)


# In[ ]:




