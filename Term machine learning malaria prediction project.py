#!/usr/bin/env python
# coding: utf-8

# # Import libraries for the analysis

# In[1]:


import sys
import shap
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import tree
from sklearn import metrics 
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error
from outlier_plotting.sns import handle_outliers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score,mean_squared_error
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,AdaBoostRegressor
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
from interpret.blackbox import LimeTabular
import warnings
warnings.filterwarnings('ignore')
shap.initjs()


# In[2]:


#read the data using pandas
Malaria_dataset = pd.read_csv("term_malaria_dataset.csv")


# ## Dataset exploration, cleaning and preprocess

# In[3]:


#find the number of rows and columns in the dataset
Malaria_dataset


# In[4]:


#find the number of rows and columns in the malaria_dataset
Malaria_dataset.shape


# In[5]:


#get the first n rows in the malaria_dataset
Malaria_dataset.head(n=5)


# In[6]:


Malaria_dataset.duplicated()


# In[7]:


# list the column names
Malaria_dataset.columns


# In[8]:


#obtain some information about the data i.e. columns,datatypes,missing values,etc
Malaria_dataset.info()


# In[9]:


#we are interested in the columns : 'Clinical_diagnosis' up to 'RBC_dist_width_Percent'
#meaning we will subset the data from column 16 - the last column
subset=Malaria_dataset.iloc[:,16:]


# In[10]:


subset.shape


# In[11]:


subset.info()


# ### Taking care of missing values in the subset

# In[12]:


#Check the mising data in each column
subset.isnull().sum()


# In[13]:


# drop all rows with missing values
subset.dropna(inplace=True)


# In[14]:


subset.shape


# In[15]:


subset.columns


# In[16]:


#getting the different malaria outcomes wich will be the labels/classes in the data
subset['Clinical_Diagnosis'].unique()


# In[17]:


labels=pd.Categorical(subset['Clinical_Diagnosis'])
labels


# In[18]:


subset.head()


# In[19]:


#class distribution
subset['Clinical_Diagnosis'].value_counts()


# In[20]:


#Checking duplicates in the subset
subset.duplicated()


# In[21]:


#Checking duplicates in the class distribution
subset['Clinical_Diagnosis'].value_counts().duplicated()


# In[22]:


# plot a bar chat to display the class distribution
sns.countplot(x="Clinical_Diagnosis",data=subset)
plt.title("Malaria class distribution", fontsize=16)


# In[23]:


# plot a bar chat to display the class distribution
#subset['Clinical_Diagnosis'].value_counts().plot.bar()
#plt.title("Malaria class distribution", fontsize=16)


# In[24]:


# Descriptive statistics on the data
subset.iloc[:,1:].describe().transpose()


# In[25]:


#check the correlation for the features
subset.corr(numeric_only = True)


# In[26]:


#lets visualize the correlation matrix using seaborn
plt.rcParams["figure.figsize"] = (15,15)
sns.heatmap(subset.corr(numeric_only = True), annot=True, linewidth=.5);


# ## Data Normalization

# In[27]:


#Normalising numerical data using pandas Gives unbiased

numerical_malaria_Dataset_columns = ['wbc_count','rbc_count','hb_level','hematocrit','mean_cell_volume','mean_corp_hb','mean_cell_hb_conc','platelet_count','platelet_distr_width','mean_platelet_vl','neutrophils_percent','lymphocytes_percent','mixed_cells_percent','neutrophils_count','lymphocytes_count','mixed_cells_count','RBC_dist_width_Percent']
numerical_malaria_Dataset = subset[numerical_malaria_Dataset_columns]
normalised_malaria_Dataset = (numerical_malaria_Dataset - numerical_malaria_Dataset.mean())/numerical_malaria_Dataset.std()
normalised_malaria_Dataset.head()


# In[28]:


# Normalising numerical data using pandas Gives unbiased (use min-max normalization)
numerical_malaria_Dataset_columns = ['wbc_count','rbc_count','hb_level','hematocrit','mean_cell_volume','mean_corp_hb','mean_cell_hb_conc','platelet_count','platelet_distr_width','mean_platelet_vl','neutrophils_percent','lymphocytes_percent','mixed_cells_percent','neutrophils_count','lymphocytes_count','mixed_cells_count','RBC_dist_width_Percent']
numerical_malaria_Dataset = subset[numerical_malaria_Dataset_columns]
numerical_malaria_Dataset = (numerical_malaria_Dataset - numerical_malaria_Dataset.min())/(numerical_malaria_Dataset.max() - numerical_malaria_Dataset.min())
numerical_malaria_Dataset.head()


# ## Correlation plots for normalized data

# In[29]:


f, x = plt.subplots(figsize=(15,15))
sns.heatmap(normalised_malaria_Dataset.corr(), annot=True, linewidth=.5, ax=x);


# In[30]:


sns.pairplot(normalised_malaria_Dataset) # pairplot for normalized asumed unbiased malaria dataset  (use min-max normalization)


# In[31]:


sns.jointplot(x=normalised_malaria_Dataset.loc[:, "rbc_count"],
             y=normalised_malaria_Dataset.loc[:, "hb_level"],
             kind="reg",
             color="green");


# ## Data Preprocessing

# In[32]:


# separate the labels/classes from the features/measurement
X=subset.iloc[:,1:]
y=subset.iloc[:,0]


# In[33]:


X.shape


# In[34]:


y.shape


# # Encoding labels

# In[35]:


#This is required by scikit learn when performing supervised learning
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder() #insatnce creation
label_encoder.fit(y)         #fitting label encoder
y_encoded=label_encoder.transform(y) # Assign encoder to class


# In[36]:


#Lets try first 5 values
y_encoded[0:6]


# In[37]:


# lets look at the original values now
y.iloc[0:6]


# In[38]:


# Knowing if the encoder is encoded in a particular class
classes=label_encoder.classes_ #getting the class
classes #print classes


# In[ ]:





# In[39]:


#seperating predictors and response
#x = subset.iloc[:,0:6]
#y = subset.iloc[:,6]


# In[ ]:





# ## Splitting data into train and test sets

# In[40]:


# train test ratio 80:20
X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2,random_state=0)


# In[ ]:





# In[41]:


#lets check the shapes
print(X.shape, X_train.shape, y.shape, y_train.shape, X_test.shape, y_test.shape)


# ## Data standardization

# In[42]:


# scale data between 0 and 1
from sklearn.preprocessing import MinMaxScaler
min_max_scaler=MinMaxScaler() # Instance min_max_scaler
X_train_scaled=min_max_scaler.fit_transform(X_train)
X_test_scaled=min_max_scaler.fit_transform(X_test)


# In[43]:


#Exploring the scaled train data
X_train_scaled[1,0]


# In[44]:


X_train.iloc[1,0]


# In[45]:


X_test_scaled[1,0]


# In[46]:


X_test.iloc[1,0]


# ## Feature Selection

# The purpose of feature selection is to select relevant features for classification. Feature selection is usually used as a pre-processing step before doing the actual learning.
# 
# Mutual information algorithm is used to compute the relevance of each feature. The top n (eg. 300) features are selected for the machine learning analysis.

# In[47]:


from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score


# ### Feature Selection using Mutual Information

# In[48]:


MI=mutual_info_classif(X_train_scaled,y_train)


# In[49]:


#select top n features. lets say 300.
#you can modify the value and see how the performance of the model changes

n_features=300
selected_scores_indices=np.argsort(MI)[::-1][0:n_features]


# In[50]:


X_train_selected=X_train_scaled[:,selected_scores_indices]
X_test_selected=X_test_scaled[:,selected_scores_indices]


# In[51]:


X_train_selected.shape


# In[52]:


X_test_selected.shape


# ## Training and testing of the models

# ## AdaBoost Classifier

# In[53]:


from sklearn.ensemble import AdaBoostClassifier


# In[54]:


Adamodel = AdaBoostClassifier(n_estimators=100,learning_rate=1)
#Adamodel.fit(X_train_scaled,y_train)
Adamodel.fit(X_train,y_train)


# In[55]:


y_pred = Adamodel.predict(X_test) 


# In[56]:


#Predictions based of Accuracy, F1_score, precision, recall and mean error bound
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%') 
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_pred)*100),'%') #error mean


# ## Naive Bayes model

# In[57]:


gnb = GaussianNB()
gnb.fit(X_train,y_train)


# In[58]:


y_pred = gnb.predict(X_test) 


# In[59]:


y_pred[0:6]


# In[60]:


#Compare original values
y_test[0:6]


# In[61]:


print(y_pred)


# In[62]:


#Predictions based of Accuracy, F1_score, precision, recall and mean error bound
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%')
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_pred)*100),'%') #error mean


# ## Training phase of Random Forest model

# In[63]:


#create random forest classifier
rfm=RandomForestClassifier()
rfm.fit(X_train,y_train)


# ### Testing RF model

# In[92]:


#create random forest classifier
rfm=RandomForestClassifier()
rfm.fit(X_train,y_train)
pred_prob = rfm.predict_proba(X_test)
y_pred = rfm.predict(X_test)


# In[93]:


pred_prob[0:6]


# In[94]:


#Compare original values
y_test[0:6]


# ## Evaluating RF model

# In[95]:


# import the metrics
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score,mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve,auc


# In[97]:


#Predictions based of Accuracy, F1_score, precision, recall and mean error bound
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%') #fit with true vales and add pred y values
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_pred)*100),'%') #error mean


# In[98]:


# Confusion matrix
confusion_matrix(y_test,y_pred)
#cm = np.array(confusion_matrix(y_test,y_pred))
#print(cm)


# In[99]:


print(classification_report(y_test,y_pred))


# ## Plot the ROC Curve

# In[100]:


from sklearn.preprocessing import label_binarize
#binarize the y_values

y_test_binarized=label_binarize(y_test,classes=np.unique(y_test))

# roc curve for classes
fpr = {}
tpr = {}
thresh ={}
roc_auc = dict()

n_class = classes.shape[0]

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:,i], pred_prob[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    # plotting    
    plt.plot(fpr[i], tpr[i], linestyle='--', 
             label='%s vs Rest (AUC=%0.2f)'%(classes[i],roc_auc[i]))

plt.plot([0,1],[0,1],'b--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='lower right')
plt.show()


# # Appying XAI

# ## Using Shapley on RF model

# In[101]:


explainer = shap.Explainer(rfm)
shap_values = explainer.shap_values(X_train)
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:],feature_names=X_train.columns)


# In[102]:


shap.decision_plot(explainer.expected_value[1], shap_values[1][0,:], feature_names=list(X_train.columns))


# In[103]:


from shap.plots import _waterfall
_waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][0,:], feature_names=list(X_train.columns))


# ## Using LIME to explain RF model

# In[104]:


import lime
from lime import lime_tabular


# In[105]:


explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train.values, 
                                                   feature_names=X_train.columns.values,verbose=True,mode='classification',
                                                   categorical_features=[0])


# In[106]:


lime_exp = explainer.explain_instance(data_row =X_test.iloc[5], predict_fn = rfm.predict_proba,num_features=17)
lime_exp.show_in_notebook(show_table=True,show_all=False)


# In[107]:


# Local explanation plot
plt = lime_exp.as_pyplot_figure()
plt.tight_layout()


# ## Logistic regression model

# In[108]:


# #create and train Logistic regression classifier
log_model = LogisticRegression(solver='liblinear', max_iter=2000)
log_model=LogisticRegression()
log_model.fit(X_train_scaled,y_train)


# In[109]:


# model prediction on the test set
y_pred=log_model.predict(X_test_scaled) #Testing phase of LR


# In[110]:


#Compare predicted values and original values
print(y_pred[1:6],y_test[1:6])


# ## Evaluation phase of the model

# In[111]:


#Predictions based of Accuracy, F1_score, precision, recall and mean error bound
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%') #fit with true vales and add pred y values
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_pred)*100),'%') #error mean


# ## Boost LR with Adaboost

# In[112]:


#Boosting LR model
mylogregmodel = LogisticRegression()
adabc =AdaBoostClassifier(n_estimators=150,base_estimator=mylogregmodel,learning_rate=1)
Adamodel = adabc.fit(X_train_scaled,y_train)
y_pred = Adamodel.predict(X_test_scaled)


# In[113]:


#Predictions based of Accuracy, F1_score, precision, recall and mean error bound
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%')
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_pred)*100),'%') #error mean


# ## Increasing LR accuracy with GridSearchCV

# In[114]:


from sklearn.model_selection import GridSearchCV
log_model = LogisticRegression()
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25,15,100]}
grid_clf_acc = GridSearchCV(log_model, param_grid = grid_values)
grid_clf_acc.fit(X_train, y_train)
#grid_clf_acc = GridSearchCV(log_model, param_grid = grid_values,scoring = 'accuracy')
#Predict values based on new parameters
y_pred = grid_clf_acc.predict(X_test)


# In[115]:


#accuracy=accuracy_score(y_test,y_pred)
#accuracy
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%')
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')


# In[116]:


# Confusion matrix
confusion_matrix(y_test,y_pred)


# In[117]:


print(classification_report(y_test,y_pred))


# ## Plots of LIME

# In[118]:


explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train.values, 
                                                   feature_names=X_train.columns.values,verbose=True,mode='classification',
                                                   categorical_features=[0])


# In[119]:


lime_exp = explainer.explain_instance(data_row =X_test.iloc[5], predict_fn = grid_clf_acc.predict_proba,num_features=20)
lime_exp.show_in_notebook(show_table=True)


# In[120]:


# Local explanation plot
plt = lime_exp.as_pyplot_figure()
plt.tight_layout()


# ## Explainable Boosting Machines (EMBs)

# ### Training Explainable Boosting Machines

# In[121]:


ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)


# In[122]:


y_preds = ebm.predict(X_test)


# In[123]:


#accuracy=accuracy_score(y_test,y_pred)
#accuracy
print('Accuracy        :',round(accuracy_score(y_test,y_preds)*100),'%')
print('f1score         :',round(f1_score(y_test,y_preds,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_preds,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_preds,average='weighted')*100),'%')


# In[124]:


# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=ebm.classes_)
disp.plot();
plt.show()


# In[125]:


ebm_global = ebm.explain_global()
show(ebm_global)


# In[126]:


ebm_local = ebm.explain_local(X_test, y_test)
show(ebm_local)


# In[127]:


cr = classification_report(y_test,y_preds)
print(cr)


# In[128]:


lime = LimeTabular(predict_fn = ebm.predict_proba, data=X_train, random_state=1)
lime_local = lime.explain_local(X_test, y_test)

show(lime_local)


# In[ ]:





# ## Decision Tree model

# In[129]:


#create and train Decision tree classifier

from sklearn.tree import DecisionTreeClassifier
treemodel=DecisionTreeClassifier()
treemodel.fit(X_train,y_train)


# In[130]:


# model prediction on the test set
y_pred=treemodel.predict(X_test) #Testing phase of DT


# In[131]:


#Compare predicted values and original values
print('predicted values',y_pred[1:6]) 
print('Real values' ,y_test[1:6])


# ## Evaluation phase of the model

# In[132]:


# import the metrics
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score,mean_squared_error
from sklearn.metrics import classification_report,confusion_matrix


# In[133]:


#Predictions based of Accuracy, F1_score, precision, recall and mean error bound
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%') #fit with true values and add pred y values
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_pred)*100),'%') #error mean


# ## Boosted Decission Tree Model

# In[134]:


#Boosting DT model
mytreemodel = DecisionTreeClassifier()
dtm =AdaBoostClassifier(n_estimators=150,base_estimator=mytreemodel,learning_rate=1)
treemodel = dtm.fit(X_train,y_train)
y_pred = treemodel.predict(X_test)


# In[135]:


#Predictions based of Accuracy, F1_score, precision, recall and mean error bound
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%')
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_pred)*100),'%') #error mean


# In[136]:


# Confusion matrix
confusion_matrix(y_test,y_pred)


# In[137]:


print(classification_report(y_test,y_pred))


# In[138]:


# Confusion matrix
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=treemodel.classes_)
disp.plot();
plt.show()


# ## Support Vector Machine model

# In[139]:


#createing and traing the SVM classifier

from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train_scaled,y_train)


# ## Testing model

# In[140]:


# model prediction on the test set
y_pred=svm.predict(X_test_scaled) #Testing phase of DT


# In[141]:


y_pred[1:6]


# In[142]:


#Compare original values
y_test[1:6]


# ## Evaluation phase of the model

# In[143]:


# import the metrics
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score,mean_squared_error
from sklearn.metrics import classification_report,confusion_matrix


# In[144]:


#Predictions based of Accuracy, F1_score, precision, recall and mean error bound
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%') #fit with true vales and add pred y values
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_pred)*100),'%') #error mean


# In[145]:


# Confusion matrix
confusion_matrix(y_test,y_pred)


# In[146]:


print(classification_report(y_test,y_pred))


# # KNN model

# In[147]:


#createing and traing the SVM classifier
knn=KNeighborsClassifier()
knn.fit(X_train_scaled,y_train)


# ## Testing model

# In[148]:


# model prediction on the test set
y_preds=knn.predict(X_test_scaled) #Testing phase of KNN


# In[149]:


y_preds[1:6]


# In[150]:


#Compare original values
y_test[1:6]


# ## Evaluation phase of the model

# In[151]:


# import the metrics
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score,mean_squared_error
from sklearn.metrics import classification_report,confusion_matrix


# In[152]:


print('Accuracy        :',round(accuracy_score(y_test,y_preds)*100),'%') #fit with true vales and add pred y values
print('f1score         :',round(f1_score(y_test,y_preds,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_preds,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_preds,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_preds)*100),'%') #error mean


# In[153]:


# Confusion matrix
confusion_matrix(y_test,y_preds)


# In[154]:


print(classification_report(y_test,y_preds))


# In[155]:


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams


# In[156]:


# Plot non-normalized and normalized confusion matrix
plt.rcParams["figure.figsize"] = (15,15)
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        knn,X_test,y_test,
        cmap=plt.cm.Reds,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


# ## K-Means Model

# In[157]:


from sklearn.cluster import KMeans


# In[158]:


x = subset.iloc[:, :].values
wcs = []  #wcss stands for 'within cluster sum of squares'

for i in range(1, 21):
    kmodel = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 250, n_init = 15, random_state = 0)
    kmodel.fit(X_train_scaled,y_train)
    wcs.append(kmodel.inertia_)

print(wcs)


# In[159]:


kmodel = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 250, n_init = 15, random_state = 0)
kmodel.fit_predict(X_train_scaled,y_train)


# In[160]:


# model prediction on the test set
y_pred=kmodel.predict(X_test_scaled) #Testing phase of K-means


# In[161]:


y


# In[162]:


y_pred[1:6]


# In[163]:


y_test[1:6]


# In[164]:


#Accuracy
print('Accuracy        :',round(accuracy_score(y_test,y_pred)*100),'%') #fit with true vales and add pred y values
print('f1score         :',round(f1_score(y_test,y_pred,average='weighted')*100),'%')
print('precision       :',round(precision_score(y_test,y_pred,average='weighted')*100),'%')
print('recall          :',round(recall_score(y_test,y_pred,average='weighted')*100),'%')
print('Mean Error bound:',round(mean_squared_error(y_test,y_pred)*100),'%') #error mean


# In[165]:


# Plotting the three clusters of first two columns(sepal length, sepal width)

#plt.scatter(subset[y == 0, 0], x[y == 1, 1], 
            #s = 50, c = 'cyan', label = 'mean_cell_volume')
#plt.scatter(x[y == 1, 0], x[y == 1, 1], 
            #s = 50, c = 'blue', label = 'mean_corp_count')
#plt.scatter(x[y == 2, 0], x[y == 2, 1],
          #  s = 50, c = 'green', label = 'RBC_dist_width_percent')

# Plotting the centroids of each clusters
#plt.scatter(kmodel.cluster_centers_[:, 0], kmodel.cluster_centers_[:,1], 
            #s = 75, c = 'red', label = 'Centroids')

#plt.legend(loc=1, bbox_to_anchor= (1.4, 1))
#plt.grid()


# ## XGBoost Regressor

# In[166]:


xg_model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10) 
# fitting the model 
xg_model.fit(X_train_scaled,y_train) 
# predict model 
y_pred = xg_model.predict(X_test_scaled) 
print(y_pred) 
#Compute the rmse by invoking the mean_sqaured_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
print("RMSE: %f" % (rmse)) 
#print('recall          :',recall_score(y_test,predictions,average='weighted'))


# In[167]:


import xgboost as xgb
xgbc_model = xgb.XGBRegressor()
xgbc_model.fit(X_train,y_train)
y_predict = xgbc_model.predict(X_test)


# In[168]:


rmse = np.sqrt(mean_squared_error(y_test, y_predict)) 
print("RMSE: %f" % (rmse)) 


# In[169]:


from sklearn.metrics import explained_variance_score


# In[170]:


print(explained_variance_score(y_predict,y_test))


# ## K-fold Cross Validation using XGBoost

# In[171]:


from xgboost import cv
data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)


# In[172]:


# label_column specifies the index of the column containing the true label
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10} 
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123) 
cv_results.head()


# In[173]:


# Extract and print the final boosting round metric 
print((cv_results["test-rmse-mean"]).head()) 


# ## Visualize Boosting Trees and Feature Importance

# In[174]:


xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10) 
xgb.plot_importance(xg_reg) 
plt.rcParams['figure.figsize'] = [5, 5] 
plt.show()


# ## XGBoost Classifier

# In[175]:


xgb_classifier = xgb.XGBClassifier()


# In[176]:


xgb_classifier.fit(X_train,y_train)


# In[177]:


# Making predictions using XGBoost
predictions = xgb_classifier.predict(X_test)
predictions


# In[178]:


print("Accuracy of Model::",accuracy_score(y_test,predictions))
print('f1score         :',f1_score(y_test,predictions,average='weighted'))
print('precision       :',precision_score(y_test,predictions,average='weighted'))
print('recall          :',recall_score(y_test,predictions,average='weighted'))


# ### Explaing XGBoost Classifier model's predictions using SHAP values

# In[179]:


# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(X_test)


# In[180]:


shap.force_plot(explainer.expected_value[1], shap_values[1][0,:],feature_names=X_train.columns)


# # Adaboost classifier

# In[181]:


crossvalidation=KFold(n_splits=10,shuffle=True,random_state=1)
for depth in range (1,10):
    tree_regressor = tree.DecisionTreeRegressor(max_depth=depth,random_state=1)
    if tree_regressor.fit(X_train_scaled,y_train).tree_.max_depth<depth:
        break
        score=np.mean(cross_val_score(tree_regressor,X_train_scaled,y_train,scoring='neg_mean_squared_error',
                     cv=crossvalidation,n_jobs=1))
        print(depth, score)


# ## Hyperparameter Tunning

# In[182]:


ada=AdaBoostRegressor()
search_grid={'n_estimators':[500,1000,2000],
            'learning_rate':[.001,0.01,.1],'random_state':[1]}
search=GridSearchCV(estimator=ada,param_grid=search_grid,
                   scoring='neg_mean_squared_error',n_jobs=1, cv=crossvalidation)
print(search_grid)
print(search)


# In[183]:


search.fit(X_train_scaled,y_train)
print(search.best_params_)
print(search.best_score_)


# In[184]:


ada2=AdaBoostRegressor(n_estimators=2000,learning_rate=0.01,random_state=1)
score=np.mean(cross_val_score(ada2,X_train_scaled,y_train,scoring='neg_mean_squared_error',
                     cv=crossvalidation,n_jobs=1))
score


# ## Determining the outliers in the dataset

# In[185]:


import plotly.express as px
#create a histogram
fig = px.histogram(Malaria_dataset, x="hematocrit")
fig.show()


# In[186]:


ax = sns.boxplot(x="Clinical_Diagnosis", y="hb_level", data=Malaria_dataset)


# ## Interquartile range

# In[187]:


def box_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
    series = series[mask]
    return series


# In[188]:


data_outliers = pd.DataFrame([])
for i in Malaria_dataset.copy().Clinical_Diagnosis.unique():
    df_outlier = Malaria_dataset.copy().sort_values(['Clinical_Diagnosis'])
    df_outlier = df_outlier.loc[df_outlier['Clinical_Diagnosis'] == i, :].reset_index(drop=True)
    df_outlier = df_outlier[['Clinical_Diagnosis', 'hb_level']].set_index('Clinical_Diagnosis').apply(box_outliers)
    df_outlier = df_outlier.reset_index()
    data_outliers = data_outliers.append(df_outlier) 


# ### These are actually the outlier values!

# In[189]:


data_outliers.reset_index(drop=True)


# # Data Normalization

# In[ ]:





# In[190]:


# Boxplot from subset data
sns.set(rc={'figure.figsize':(13,5)})
#create seaborn boxplots by group
subset = pd.DataFrame(data = np.random.random(size=(5,9)), 
        columns = ['platelet_distr_width','mean_platelet_vl','neutrophils_percent','lymphocytes_percent','mixed_cells_percent','neutrophils_count','lymphocytes_count','mixed_cells_count','RBC_dist_width_Percent'])
sns.boxplot(x="variable", y="value", data=pd.melt(subset))
plt.show()


# In[191]:


#=['wbc_count','rbc_count','hb_level','hematocrit','mean_cell_volume','mean_corp_hb','mean_cell_hb_conc','platelet_count','platelet_distr_width','mean_platelet_vl','neutrophils_percent','lymphocytes_percent','mixed_cells_percent','neutrophils_count','lymphocytes_count','mixed_cells_count','RBC_dist_width_Percent']


# ## Correlation plots for raw dataset

# In[192]:


corr = Malaria_dataset.corr()
sns.heatmap(corr,vmin=-1, vmax=1,annot=True,linewidth=.5)
#sns.set(rc={'figure.figsize':(16,16)})


# In[ ]:




