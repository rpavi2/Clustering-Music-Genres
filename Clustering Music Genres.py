#!/usr/bin/env python
# coding: utf-8

# # Clustering Music Genres (Problem Statement)
# 
# Every person has a different taste in music. We cannot identify what kind of music does a person likes by just knowing about their lifestyle, hobbies, or profession. So it is difficult for music streaming applications to recommend music to a person. But if we know what kind of songs a person listens to daily, we can find similarities in all the music files and recommend similar music to the person.
# 
# That is where the cluster analysis of music genres comes in. Here you are given a dataset of popular songs on Spotify, which contains artists and music names with all audio characteristics of each music. Our goal is to group music genres based on similarities in their audio characteristics.

# In[1]:


(/Start, with, this, task, by, importing, the, necessary, Python, libraries, and, the, dataset)

import pandas as pd
import numpy as np
from sklearn import cluster

data = pd.read_csv("Spotify-2000.csv")
print(data.head())


# In[2]:


data = data.drop("Index", axis=1)


# ## The correlation between all the audio features in the dataset

# In[9]:


print(data.corr())


# ## Clustering Analysis of Audio Features

# Use the K-means clustering algorithm to find the similarities between all the audio features. Then I will add clusters in the dataset based on the similarities we found. So let’s create a new dataset of all the audio characteristics and perform clustering analysis using the K-means clustering algorithm:

# In[10]:


data2 = data[["Beats Per Minute (BPM)", "Loudness (dB)", 
              "Liveness", "Valence", "Acousticness", 
              "Speechiness"]]

from sklearn.preprocessing import MinMaxScaler
for i in data.columns:
    MinMaxScaler(i)
    
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(data2)


# #### Now let's add the clusters as predicted by the K-means clustering algorithm to the original dataset:

# In[11]:


data["Music Segments"] = clusters
MinMaxScaler(data["Music Segments"])
data["Music Segments"] = data["Music Segments"].map({1: "Cluster 1", 2: 
    "Cluster 2", 3: "Cluster 3", 4: "Cluster 4", 5: "Cluster 5", 
    6: "Cluster 6", 7: "Cluster 7", 8: "Cluster 8", 
    9: "Cluster 9", 10: "Cluster 10"})


# In[12]:


print(data.head())


# #### let’s visualize the clusters based on some of the audio features:

# In[14]:


import plotly.graph_objects as go
PLOT = go.Figure()
for i in list(data["Music Segments"].unique()):
    

    PLOT.add_trace(go.Scatter3d(x = data[data["Music Segments"]== i]['Beats Per Minute (BPM)'],
                                y = data[data["Music Segments"] == i]['Energy'],
                                z = data[data["Music Segments"] == i]['Danceability'],                        
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(i)))
PLOT.update_traces(hovertemplate='Beats Per Minute (BPM): %{x} <br>Energy: %{y} <br>Danceability: %{z}')

    
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'Beats Per Minute (BPM)', titlefont_color = 'black'),
                                yaxis=dict(title = 'Energy', titlefont_color = 'black'),
                                zaxis=dict(title = 'Danceability', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color  = 'black', size = 12))


# ## Conclusion 

# Here we can anlayse the cluster analysis of music genres with machine learning using Python. 
# Clustering music genres is a task of grouping music based on the similarities in their audio features. 

# In[ ]:




