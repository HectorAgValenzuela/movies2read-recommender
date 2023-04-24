import re
import pandas as pd 
import numpy as np

#Function to import the data from the csv file
def importData(filename):
    #fileName = "Data\steam.csv"
    df = pd.read_csv(filename, encoding = "utf-8")
    return df

#Function to fill all Rating NAn values for a an averge of 5

def filldummyRating(df):
    df['RatingDesc'] == 5
    return df
#Function to fill all Rating NAn values for a an averge of 3


def fillnullRating(df):
    df['Rating'].fillna(value=3)
    return df

#Function to get the total amount of ratings

def totalRatings(row):
    posCount = row['Ratings']
    negCount = row['RatingsDesc']
    totalCount = posCount + negCount
    return totalCount
#Function to create the average Score of the ratings

def createScore(row):
    posCount = row['Ratings']
    negCount = row['RatingsDesc']
    totalCount = posCount + negCount
    average = posCount / totalCount
    return round(average, 2)
#Function to add the score and total ratings to the dataframe

def addScoreAndTotalRatings(df):
    df['score'] = df.apply(createScore, axis=1)
    df['total_ratings'] = df.apply(totalRatings, axis=1)
    return df
#funcion viendo si se queda     
#def replace_foreign_characters(s):
 #   return re.sub(r'[^\x00-\x7f]',r'', s)

# Function that computes the weighted rating of each game
def weighted_rating(x, m, C):
    v = x['total_ratings']
    R = x['score']
    # Calculation based on the IMDB formula
    return round((v/(v+m) * R) + (m/(m+v) * C), 2)

def addWeightedRating(df):
    # Calculate mean of vote average column
    C = df['score'].mean()
    # Calculate the minimum number of votes required to be in the chart
    m = df['total_ratings'].quantile(0.90)
    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    df['weighted_score'] = df.apply(weighted_rating, axis=1, args=(m, C))
    return df

def combine(x, *features):
    result = ''
    for f in features:
        result += str(x[f]) + ' '
    return result

# Funcion que combina las  .
def formatColumns(df):
    features = ['BodyPart','Equipment', 'Level', 'Type']
    #Compbine the features into one column
    df['merged'] = df.apply(combine, axis=1, args = features)
    return df
    