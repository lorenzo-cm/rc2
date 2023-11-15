import os
import pandas as pd
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

    df_targets['Rating'] = [pred.est for pred in predictions]
    
    if not os.path.exists('results'):
        os.makedirs('results')

    df_targets.to_csv('results/target_predictions.csv', index=False)
    
        # Sort the DataFrame by UserId and then by Rating in descending order
    df_sorted = df_targets.sort_values(by=['UserId', 'Rating'], ascending=[True, False])

    # Drop the Rating column as it's not needed in the final output
    df_final = df_sorted.drop('Rating', axis=1)

    # Write to a CSV file
    df_final.to_csv('sorted_items_per_user.csv', index=False)