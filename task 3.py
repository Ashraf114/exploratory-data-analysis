#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from IPython.display import Image


# In[7]:


iris = load_iris()
X=iris.data[:,:] 
y=iris.target


# In[8]:


data=pd.DataFrame(iris['data'],columns=["Petal length","Petal Width","Sepal Length","Sepal Width"])
data['Species']=iris['target']
data['Species']=data['Species'].apply(lambda x: iris['target_names'][x])

data.head()


# In[9]:


data.shape


# In[10]:


data.describe()


# In[11]:


sns.pairplot(data)


# In[12]:


sns.FacetGrid(data,hue='Species').map(plt.scatter,'Sepal Length','Sepal Width').add_legend()
plt.show()


# In[13]:


sns.FacetGrid(data,hue='Species').map(plt.scatter,'Petal length','Petal Width').add_legend()
plt.show()


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) 
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train,y_train)
print("Training Complete.")
y_pred = tree_classifier.predict(X_test)


# In[17]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
df


# In[18]:


print("Class Names = ",iris.target_names)

# Estimating class probabilities
print()
print("Estimating Class Probabilities for flower whose petals length width are 4.7cm and 3.2cm and sepal length and width are 1.3cm and 0.2cm. ")
print()
print('Output = ',tree_classifier.predict([[4.7, 3.2, 1.3, 0.2]]))
print()
print("Our model predicts the class as 0, that is, setosa.")


# In[19]:


print("Accuracy:",sm.accuracy_score(y_test, y_pred))


# The accuracy of this model is 1 or 100% since I have taken all the 4 features of the iris dataset for creating the decision tree model.
