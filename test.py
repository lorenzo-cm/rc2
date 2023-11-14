import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
import pickle
import os
import argparse


# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('--new_model', action='store_true', 
                    help='Indicates if a new model should be used')

parser.add_argument('--train_all', action='store_true', 
                    help='Indicates to use all data to train')

parser.add_argument('--targets', action='store_true', 
                    help='Indicates to generate the predictions for the targets')

# Parse the arguments
args = parser.parse_args()


# Assuming 'transformed_records' is your dataset after transformation
df_ratings = pd.read_json('data/ratings.jsonl', lines=True)

# Define a reader with the rating scale
reader = Reader(rating_scale=(min(df_ratings['Rating']), max(df_ratings['Rating'])))

# Load the dataset into Surprise
data = Dataset.load_from_df(df_ratings[['UserId', 'ItemId', 'Rating']], reader)


if os.path.exists('trained_model.pkl') and not args.new_model:
    print('Loading model from file')
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)

else:
    print('Training new model')
    
    test_size = 0.2

    if args.train_all:
        print('Using all data to train')
        test_size = 0
        
    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Use the SVD algorithm
    model = SVD()

    # Train the model
    model.fit(train_data)

    #save the model
    model_filename = 'trained_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    print(f'Model saved to {model_filename}')


if not args.train_all:
    print('Evaluating model')
    
    # Make predictions on the test set
    predictions = model.test(test_data)

    # Calculate and print RMSE and MAE
    rmse_value = rmse(predictions, verbose=False)
    mae_value = mae(predictions, verbose=False)

    for pred in predictions[:10]:
        print(pred)

    # Output RMSE and MAE values
    print(f'RMSE: {rmse_value}, MAE: {mae_value}')


if args.targets:
    print('Generating predictions for targets')
    
    df_targets = pd.read_csv('data/targets.csv')
    
    # Convert df_targets to a list of tuples for prediction
    test_data = [(row['UserId'], row['ItemId'], None) for index, row in df_targets.iterrows()]

    # Make predictions
    predictions = [model.predict(uid, iid, r_ui=verdict, verbose=False) for (uid, iid, verdict) in test_data]

    df_targets['Rating'] = [pred.est for pred in predictions]
    
    df_targets.to_csv('data/target_predictions.csv', index=False)
    
        # Sort the DataFrame by UserId and then by Rating in descending order
    df_sorted = df_targets.sort_values(by=['UserId', 'Rating'], ascending=[True, False])

    # Drop the Rating column as it's not needed in the final output
    df_final = df_sorted.drop('Rating', axis=1)

    # Write to a CSV file
    df_final.to_csv('sorted_items_per_user.csv', index=False)