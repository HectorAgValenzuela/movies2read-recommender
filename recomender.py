from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import joblib
import pickle
import Prepross
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

import json

app = Flask(__name__)
api = Api(app)
CORS(app)


df_recipes = pd.read_csv("Book_P_Path.csv")
moovieData = pd.read_csv("Movie_P_Path.csv")
bookData = pd.read_csv("Book_P_Path.csv")
bookDataParse = bookData["Tags"].values.astype('U')

        
nweTfidf = joblib.load("./TFIDF_MODEL_PATH.pkl") 
tfidf_recipe = joblib.load("./TFIDF_ENCODING_PATH.pkl")

def get_recommendations(N, scores ):
    # order the scores with and filter to get the highest N scores

    top = sorted(range(len(scores)), key= (lambda i: scores[i]), reverse=True)
    #sorted_indexes = np.argsort(scores[0])[::-1]
    # create dataframe to load in recommendations
    return json.dumps(
        df_recipes.iloc[top].values[:N].tolist())

def Recomendar(movieID):
    

    movies_tfidf = nweTfidf.transform(moovieData[ moovieData["item_id"] == int(movieID)]["Tags"])
    cos_sim = map(lambda x: cosine_similarity(movies_tfidf, x), tfidf_recipe)
    scores = list(cos_sim)
    aux = get_recommendations(5, scores)
    return aux
    
class status (Resource):
    def get(self):
        try:
            return {'data': 'Api is Running'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}


class Recommender(Resource):
    def get(self):
        books = request.args.get('books')
        print(books)
        return jsonify({'data': Recomendar(books)})
            

#api.add_resource(status, '/')
#api.add_resource(Recommender, '/recommender')

Recomendar("1")

if __name__ == '__main__':
       # Apply the PreProcess functions to the dataset
   dataDF = Prepross.importData("data\megaGymDataset.csv")
   print("Before: " + str(dataDF.shape))
   dataDF = Prepross.filldummyRating(dataDF)
   dataDF = Prepross.filldummyRating(dataDF)
   print("After: " + str(dataDF.shape))
   dataDF = Prepross.addScoreAndTotalRatings(dataDF)
   dataDF = Prepross.addWeightedRating(dataDF)
   dataDF = Prepross.formatColumns(dataDF)
   print("After: " + str(dataDF.shape))

   
   #Export the processed data to a csv file
   dataDF.to_csv('data\cleanedData.csv', index=False)
   
   # create an object for TfidfVectorizer
   tfidfVector = TfidfVectorizer(stop_words='english')

   # convert the list of documents (rows of features) into a matrix
   tfidfMatrix = tfidfVector.fit_transform(dataDF['merged'])
   print(tfidfMatrix.shape)
   # create the cosine similarity matrix
   sim_matrix = cosine_similarity(tfidfMatrix,tfidfMatrix)\
      
   print(sim_matrix.shape)
   print(sim_matrix)
   


api.add_resource(status, '/')
api.add_resource(Recommender, '/recommender')

Recomendar("1")

app.run()