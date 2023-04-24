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


def get_appid_from_index(df, index):
   return df[df.index == index]['appid'].values[0]
def get_title_year_from_index(df, index):
   return df[df.index == index]['year'].values[0]
def get_title_from_index(df, index):
   return df[df.index == index]['name'].values[0]
def get_index_from_title(df, title):
   return df[df.name == title].index.values[0]
def get_score_from_index(df, index):
   return df[df.index == index]['score'].values[0]
def get_weighted_score_from_index(df, index):
   return df[df.index == index]['weighted_score'].values[0]
def get_total_ratings_from_index(df, index):
   return df[df.index == index]['total_ratings'].values[0]
def get_platform_from_index(df, index):
   return df[df.index == index]['platforms'].values[0]



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

#_________________________________________________________________________________________________________________
def recommend(df, how_many, game_name, sort_option, min_year, platform, min_score, sm_matrix):
   #Create a Dataframe with these column headers
   recomm_df = pd.DataFrame(columns=['Game Title', 'Year', 'Score', 'Weighted Score', 'Total Ratings'])
   #find the corresponding index of the game title
   games_index = get_index_from_title(df, game_name)
   #return a list of the most similar game indexes as a list
   games_list = list(enumerate(sm_matrix[int(games_index)]))
   #Sort list of similar games from top to bottom
   similar_games = list(filter(lambda x:x[0] != int(games_index), sorted(games_list,key=lambda x:x[1], reverse=True)))
   #Print the game title the similarity matrix is based on
   print('Here\'s the list of games similar to ' + str(game_name) + ':\n')
   #Only return the games that are on selected platform
   n_games = []
   for i,s in similar_games:
      if platform in get_platform_from_index(df, i):
         n_games.append((i,s))
   #Only return the games that are above the minimum score
   high_scores = []
   for i,s in n_games:
      if get_score_from_index(df, i) > min_score:
         high_scores.append((i,s))
   n_games_min_years = []        
   for i,s in n_games:
      if get_title_year_from_index(df, i) >= min_year:
         n_games_min_years.append((i,s))
   #Return the game tuple (game index, game distance score) and store in a dataframe
   for i,s in n_games_min_years[:how_many]:
      #Dataframe will contain attributes based on game index
      row = pd.DataFrame({'Game Title': get_title_from_index(df, i), 'Year': get_title_year_from_index(df, i), 'Score': get_score_from_index(df, i), 'Weighted Score': get_weighted_score_from_index(df, i), 'Total Ratings': get_total_ratings_from_index(df,i)}, index = [0])
      #Append each row to this dataframe
      recomm_df = pd.concat([row, recomm_df])
   #Sort dataframe by Sort_Option provided by 
   recomm_df = recomm_df.sort_values(sort_option, ascending=False)
   #Only include games released same or after minimum year 
   recomm_df = recomm_df[recomm_df['Year'] >= min_year]
   return recomm_df

#_____________________________________________________________________________
def Recomendar(movieID):
    
    movies_tfidf = nweTfidf.transform(moovieData[ moovieData["item_id"] == int(movieID)]["Tags"])
    cos_sim = map(lambda x: cosine_similarity(movies_tfidf, x), tfidf_recipe)
    scores = list(cos_sim)
    aux = get_recommendations(5, scores)
    return aux
#___________________________________________________________________________________________________

class status (Resource):
    def get(self):
        try:
            return {'data': 'Api is Running'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}
#______________________________________________________________________________________________
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