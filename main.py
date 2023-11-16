import os
import pandas as pd
import numpy as np
from surprise.model_selection import train_test_split

import src.parse_args as parse_args
import src.data_loader as data_loader
import src.model as SVD_model

# global variables
test_size = 0.2
random_state = 42

# Parse args
args = parse_args.parse()

# Load data
data = data_loader.load_ratings()
df_content = data_loader.load_content()

# Divide into data into test and train
if args.train_all:
    print('Using all data to train')
    test_size = 0
    train_data = data.build_full_trainset()
else:
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)


# model
model = SVD_model.RecommenderSVD()

if not args.new_model:
    print('Loading model from file')
    model.load_model('model/trained_model.pkl')

else:
    print('Training new model')
    model.train(train_data)

    model_filename = 'model/trained_model.pkl'
    model.save_model(model_filename)
    print(f'Model saved to {model_filename}')


if not args.train_all:
    print('Evaluating model')
    preds, rmse_value, mae_value = model.test(test_data)

    for pred in preds[:10]:
        print(pred)

    print(f'RMSE: {rmse_value}, MAE: {mae_value}')


if args.targets:
    print('Generating predictions for targets')
    
    df_targets = pd.read_csv('data/targets.csv')
    
    # Convert df_targets to a list of tuples for prediction
    test_data = list(zip(df_targets['UserId'], df_targets['ItemId'], [None] * len(df_targets)))

    # Make predictions
    predictions = [model.model.predict(uid, iid, r_ui=verdict, verbose=False) for (uid, iid, verdict) in test_data]
    model_preds = np.array([pred.est for pred in predictions])

    # substitute nan with mean
    means = df_content.mean(numeric_only=True)
    df_content2 = df_content.fillna(means)

    # lookup item info
    lookup_table = df_content2.set_index('ItemId')[['imdbVotes', 'Metascore', 'rtRating', 'imdbRating']].to_dict(orient='index')

    # The final rating will be a weighted sum of some features
    final_preds = []
    for i in range(len(model_preds)):
        itemId = predictions[i].iid
        item_info = lookup_table[itemId]
        rating = 0.25 * model_preds[i] + \
                 0.6 * item_info['imdbVotes'] + \
                 0.05 * item_info['Metascore'] + \
                 0.05 * item_info['rtRating'] + \
                 0.05 * item_info['imdbRating']
        np.clip(rating, 0, 10)
        final_preds.append(rating)

    df_targets['Rating'] = final_preds
    
    if not os.path.exists('results'):
        os.makedirs('results')

    df_targets.to_csv('results/target_predictions.csv', index=False)
    
        # Sort the DataFrame by UserId and then by Rating in descending order
    df_sorted = df_targets.sort_values(by=['UserId', 'Rating'], ascending=[True, False])

    # Drop the Rating column as it's not needed in the final output
    df_final = df_sorted.drop('Rating', axis=1)

    # Write to a CSV file
    df_final.to_csv('sorted_items_per_user.csv', index=False)