import pandas as pd
import numpy as np

from surprise import Dataset, Reader

def load_ratings():
    df_ratings = pd.read_json('data/ratings.jsonl', lines=True)

    # Define a reader with the rating scale
    reader = Reader(rating_scale=(min(df_ratings['Rating']), max(df_ratings['Rating'])))

    # Load the dataset into Surprise
    return  Dataset.load_from_df(df_ratings[['UserId', 'ItemId', 'Rating']], reader)

def load_content():
    df_content = pd.read_json('data/content.jsonl', lines=True)

    # getting the rotten tomatoes ratings
    rt_ratings = []
    for ratings_list in df_content['Ratings']:
        rt_rating = next((item['Value'] for item in ratings_list if item['Source'] == 'Rotten Tomatoes'), None)
        if rt_rating:
            rt_rating = int(rt_rating[:-1])
        rt_ratings.append(rt_rating)
    df_content['rtRating'] = rt_ratings

    # getting useful columns
    data_content = df_content[['ItemId', 'Metascore', 'imdbRating', 'imdbVotes', 'rtRating']]

    # replacing string 'N/A' to np.nan and , number separator
    data_content = data_content.replace('N/A', np.nan)
    data_content['imdbVotes'] = data_content['imdbVotes'].str.replace(',', '')

    # converting to numeric data
    data_content['Metascore'] = data_content['Metascore'].astype('Int32')
    data_content['imdbRating'] = data_content['imdbRating'].astype('float32')
    data_content['imdbVotes'] = data_content['imdbVotes'].astype('Int32')
    
    return data_content