#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 


# In[2]:


wine = pd.read_csv("wine.csv")


# In[3]:


wine.head()


# In[4]:


wine.shape


# In[5]:


wine.info()


# In[6]:


wine_df = wine.iloc[:,1:]


# In[7]:


wine_df


# In[8]:


wine_df.corr()


# In[9]:


win = wine_df.values


# In[10]:


win


# In[11]:


wine_norm = scale(win)


# In[12]:


wine_norm


# In[13]:


pca = PCA()
pca_values = pca.fit_transform(wine_norm)


# In[14]:


pca_values


# In[15]:


var = pca.explained_variance_ratio_
var


# In[16]:


var_c = np.cumsum(np.round(var, decimals = 4)*100)
var_c


# In[17]:


pca.components_


# In[18]:


plt.plot(var_c, color = 'red')


# In[19]:


a = pca_values[:,0:1]
b = pca_values[:,1:2]
c = pca_values[:,2:3]


# In[20]:


a


# In[21]:


b


# In[22]:


c


# In[23]:


plt.scatter(x = a, y= b)
plt.xlabel("pc1")
plt.ylabel("pc2")


# In[25]:


final_df = pd.concat([pd.DataFrame(pca_values[:,0:3], columns = ['pc1','pc2','pc3']), wine[['Type']]], axis = 1)
final_df


# In[26]:


import seaborn as sns


# In[27]:


plt.figure(figsize = (10,8))
sns.scatterplot(data = final_df, x = 'pc1', y = 'pc2', hue = 'Type')


# In[28]:


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[29]:


final_df1 = final_df.iloc[:,0:3]
final_df1


# In[30]:


plt.figure(figsize = (10,8))

dendrogram = sch.dendrogram(sch.linkage(final_df1, method = 'complete'))


# In[31]:


hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')


# In[32]:


y_hc = hc.fit_predict(wine_norm)
y_hc


# In[33]:


cluster_h = pd.DataFrame(y_hc, columns = ['cluster_h'])


# In[34]:


cluster_h


# In[35]:


wine['cluster_h'] = hc.labels_


# In[36]:


wine


# In[37]:


wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final_df1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Elbow curve")
plt.xlabel("K values")
plt.ylabel("wcss")
plt.show()


# In[38]:


model_k = KMeans(n_clusters =4 )
model_k.fit(wine_norm)


# In[39]:


labels = model_k.labels_
labels


# In[40]:


md = pd.Series(model_k.labels_)
wine['cluster_K'] = md


# In[41]:


wine


# In[ ]:




