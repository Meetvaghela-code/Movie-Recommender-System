import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# print(movies.head())
# print(credits.head())

movies = movies.merge(credits, on='title')

# print(movies.head().shape)

#genres,id, keywords, title, overview, cast, crew keeping column genres,id, keywords, title, overview, cast, crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# print(movies.head())
# print(movies.isnull().sum())

movies.dropna(inplace=True) #remove null values

# print(movies.isnull().sum())
# print(movies.duplicated().sum())

# print(movies.iloc[0].genres)

def convert(obj):
    list = []
    for i in ast.literal_eval(obj):
        list.append(i['name'])
    return list
# print(convert([{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]))

movies['genres'] = movies['genres'].apply(convert)

# print(movies.head())

movies['keywords'] = movies['keywords'].apply(convert)

# print(movies.head())

def convert3(obj):
    list = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            list.append(i['name'])
            count += 1
        else:
            break
    return list

movies['cast'] = movies['cast'].apply(convert3)

# print(movies.head())

def fetch_director(obj):
    list = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            list.append(i['name'])
            break
    return list

movies['crew'] = movies['crew'].apply(fetch_director)

# print(movies.head())


movies['overview'] = movies['overview'].apply(lambda x: x.split()) #string to list conversion

# print(movies.head())
#remove spaces from the list
movies['genres'] = movies['genres'].apply(lambda x : [i.replace(' ', '') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(' ', '') for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(' ', '') for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(' ', '') for i in x])

#concatenate all the lists and create a new column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# print(movies.head())


df = movies[['movie_id','title','tags']]

print(df.head())
df['tags'] = df['tags'].apply(lambda x :"".join(x)) #convert list to string

# print(df.head())
df['tags'] = df['tags'].apply(lambda x : x.lower()) #convert to lower case

# print(df.head())
ps = PorterStemmer() #such as a action and actions convert into action

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

df['tags'] = df['tags'].apply(stem)
# print(df['tags'])

#vectorization

cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(df['tags']).toarray()

# print(vectors)

# print(cv.get_feature_names_out()) # giving a single word from the vector

cosine_sim = cosine_similarity(vectors)

# print(cosine_sim)

# print(cosine_sim[0])

# print(cosine_sim[0].shape)

def recommend(movie):
    movie_index = df[df['title'] == movie].index[0]
    distances = cosine_sim[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:10]
    for i in movies_list:
        print(df.iloc[i[0]].title)

recommend('Batman Begins')

pickle.dump(df.to_dict(),open('movies_dict.pkl','wb'))
pickle.dump(cosine_sim,open('cosine_sim.pkl','wb'))