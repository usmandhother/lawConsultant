#!/usr/bin/env python
# encoding: utf-8
import json
from json import JSONEncoder

# In[252]:


import numpy as np
# linear algebra
import pandas as pd
import os

# data processing, CSV file I/O (e.g. pd.read_csv)


# In[253]:


df = pd.read_csv('updated_dataset_Lawyers.csv')

# In[254]:


df.head()

# In[255]:


df.dropna()

# In[256]:


for value in df['LawFields']:
    print(type(value))

# In[257]:


# Convert the column to string data type
df['LawFields'] = df['LawFields'].astype(str)

# In[258]:


df['tags'] = df['Ratings'].astype(str).str.cat(
    [df['CasesWon'], df['YearsofExperience'], df['LawFields']],
    sep=' ')

# In[319]:


df['tags'] = df['tags'].apply(lambda x: x.lower())
df['tags']

# In[329]:


df.nunique()

# In[321]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=2000, stop_words='english')

# In[322]:


vectors = cv.fit_transform(df['tags']).toarray()

# In[323]:


vectors

# In[324]:


vectors[0]

# In[325]:


vectors.shape

# In[326]:


cv.get_feature_names_out()

# In[267]:


from sklearn.metrics.pairwise import cosine_similarity

# In[268]:


similarity = cosine_similarity(vectors)

# In[269]:


similarity[1]

# In[345]:

recomended_lawyer = []
df['ID'] = df['ID'].astype(str)


def recommend(cases):
    index = df[df['CasesWon'] == cases].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        global recomended_lawyer
        recomended_lawyer.append(dict(df.iloc[i[0]]))


# In[347]:


# recommend('80CasesWon')

# In[348]:


# In[349]:
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def main():
   return jsonify({"message": "working", "code": 0})


@app.route("/recommend", methods=["POST"])
def index():
    # Get the input from the user
    input_string = request.form.get("input")

    # Return a list of items that match the input
    recommend(input_string)
    print(recomended_lawyer[0])
    response = jsonify({"lawyers": recomended_lawyer});
    response.headers.add('Content-Type', 'application/json')

    return response


if __name__ == "__main__":
    app.run(port=os.getenv("PORT", default=5000))

# In[ ]:
