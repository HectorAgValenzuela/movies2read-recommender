from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

import json

app = Flask(__name__)
api = Api(app)
CORS(app)

df_recipes = pd.read_csv("Book_P_Path.csv")


def get_recommendations(N, scores ):
    # load in recipe dataset
    df_recipes.set_index("item_id")
    # order the scores with and filter to get the highest N scores

    top = sorted(range(len(scores)), key= (lambda i: scores[i]), reverse=True)
    #sorted_indexes = np.argsort(scores[0])[::-1]
    # create dataframe to load in recommendations
    return json.dumps(
        df_recipes.iloc[top].values[:N].tolist())

def Recomendar(movieID):
    moovieData = pd.read_csv("Movie_P_Path.csv")
    bookData = pd.read_csv("Book_P_Path.csv")
    bookDataParse = bookData["Tags"].values.astype('U')
    nweTfidf = TfidfVectorizer() 
    tfidf_recipe = nweTfidf.fit_transform(bookDataParse)

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
            

api.add_resource(status, '/')
api.add_resource(Recommender, '/recommender')

if __name__ == '__main__':

    app.run()