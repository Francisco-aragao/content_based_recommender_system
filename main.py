# execution: python3 main.py ratings.csv targets.csv

import argparse # default python library

import pandas as pd
import json

from sklearn.model_selection import train_test_split

from surprise import SVD, Reader, Dataset, model_selection, accuracy, dump

TEST_SIZE = 0.15
RANDOM_STATE = 0  # ensure reproducibility
N_FACTORS = 120 
N_EPOCHS = 15 
BIAS = True # this set to use (or not) the user and item bias in the model

def receive_args() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Receive the arguments from the command line
    """
    
    parser = argparse.ArgumentParser(description="""
        Process input files to make recomendations. 
        Run with: [ python3 main.py ratings.jsonl content.jsonl targets.csv > submission.csv ]                                 
    """)

    parser.add_argument("ratings", type=str, help="ratings jsonl file")
    parser.add_argument("content", type=str, help="targets jsonl file")
    parser.add_argument("targets", type=str, help="targets CSV file")
    parser.add_argument("--storeOutput", type=str, help="flag to store output in a csv file")
    parser.add_argument("--usingStoredModel", type=str, help="flag to read a stored model from the path given")

    args = parser.parse_args()

    if not (args.ratings and args.targets and args.content):
        raise Exception

    # load the jsonl data using pandas
    ratings = pd.read_json(args.ratings, lines=True)
    content = pd.read_json(args.content, lines=True)

    """ with open(args.ratings, 'r', encoding='utf-8') as file:
        ratings = [json.loads(line.strip()) for line in file]
    
    with open(args.content, 'r', encoding='utf-8') as file:
        content = [json.loads(line.strip()) for line in file] """

    targets = pd.read_csv(args.targets)

    store_output = args.storeOutput if args.storeOutput else None

    load_model = args.usingStoredModel if args.usingStoredModel else None

    return ratings, content, targets, store_output, load_model


if __name__ == "__main__":

    # receive parameters
    ratings, content, targets, store_output, load_model = receive_args()

    if load_model:
        # load the model
        pass

    # print len ratings
    print("Ratings length: ", len(ratings))

    # calculating matrix sparsity
    #sparsity = 1 - len(ratings) / (len(ratings["UserId"].unique()) * len(ratings["ItemId"].unique()))
    # print("Matrix Sparsity: ", sparsity)

    # create the model
    reader = Reader(rating_scale=(ratings.Rating.min(), ratings.Rating.max()))
    data = Dataset.load_from_df(ratings[['UserId', 'ItemId', 'Rating']], reader)

    trainset, testset = model_selection.train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # print len train_ratings


    # Train FunkSVD
    model = SVD(n_factors=N_FACTORS, n_epochs=N_EPOCHS, biased=True, random_state=RANDOM_STATE)
    model.fit(trainset)

    # store the model
    dump.dump('model.pkl', algo=model)

    # Make predictions on the test set
    test_predictions = model.test(testset)

    # Extract the predictions
    test_ratings = [(pred.uid, pred.iid, pred.est) for pred in test_predictions]

    # evaluate the model

    rmse = accuracy.rmse(test_predictions)
    mae = accuracy.mae(test_predictions)

    target_predictions = []
    for row in targets.itertuples():
        uid = row.UserId
        iid = row.ItemId
        pred = model.predict(uid, iid).est
        target_predictions.append(pred)

    # merge the predictions with the targets in one dataframe
    targets['PredictedRating'] = target_predictions

    # sort the predictions based on the rating
    targets = targets.sort_values(by='PredictedRating', ascending=False)

    # remove the PredictedRating column
    targets = targets.drop(columns=['PredictedRating'])

    # store targets
    if store_output:
        targets.to_csv(store_output, index=False)
    else:
        print(targets)
