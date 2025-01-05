from content_recomendation import ContentRecomendation

import argparse
import sys
import numpy as np
import pandas as pd

# TRAIN PARAMS
FACTORS = 150
EPOCHS = 25
LR = 0.005 
REG = 0.02
USE_BIAS = True

LEARNING_RATE_WEIGHTS = 0.005
GRADIENT_EPOCHS = 10


def initParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""Recommender System with learned weights using Gradient Descent.
        Usage: python3 main.py ratings.jsonl content.jsonl targets.csv > submission.csv"""
    )

    parser.add_argument(
        "ratings",
        metavar="ratings",
        type=str,
        help="JSONL file with user-item-timestamp ratings to train the model",
    )

    parser.add_argument(
        "content",
        metavar="content",
        type=str,
        help="JSONL file with item contents to train the model",
    )

    parser.add_argument(
        "targets",
        metavar="targets",
        type=str,
        help="CSV file with user-item pairs to predict",
    )
    parser.add_argument("--storeOutput", type=str, help="flag to store output in a csv file")

    return parser




if __name__ == "__main__":
    parser = initParser()
    args = parser.parse_args()

    contentRecomendation = ContentRecomendation()

    # Load data
    data, dfRatings = contentRecomendation.loadRatings(args.ratings)
    dfContent = contentRecomendation.loadContent(args.content)

    """ trainingData = data.build_full_trainset()

    # Train the model
    model = SVD(
        n_factors=FACTORS,
        n_epochs=EPOCHS,
        lr_all=LR,
        reg_all=REG,
        biased=USE_BIAS,
    )
    model.fit(trainingData) """

    model = contentRecomendation.train_SVD_model(data, FACTORS, EPOCHS, LR, REG, USE_BIAS)

    # Generate predictions
    dfTargets = pd.read_csv(args.targets)
    testData = list(
        zip(dfTargets["UserId"], dfTargets["ItemId"], [None] * len(dfTargets))
    )
    predictions = [
        model.predict(uid, iid, r_ui=None, verbose=False) for (uid, iid, _) in testData
    ]

    fullPredictions = np.array([pred.est for pred in predictions])

    # Lookup item info
    itemLUT = dfContent.set_index("ItemId")[
        ["imdbVotes", "Metascore", "rtRating", "imdbRating", "Awards"]
    ].to_dict(orient="index")

    weights = contentRecomendation.initialize_weights()

    # Trainable weights
    """ weights = {
        "prediction": 0.3,
        "imdb_votes": 0.00001,
        "metascore": 0.3,
        "rt_rating": 0.39998,
        "imdb": 0.00001,
        "bias_awards": 2.0,
    } """

    # Prepare training data
    train_data = [
        (row["UserId"], row["ItemId"], row["Rating"])
        for _, row in dfRatings.iterrows()
    ]

    # Perform gradient descent to learn weights
    weights = contentRecomendation.gradient_descent(weights, train_data, itemLUT, fullPredictions, lr=LEARNING_RATE_WEIGHTS, epochs=GRADIENT_EPOCHS)

    # Make final predictions
    finalPredictions = []
    for i in range(len(fullPredictions)):
        itemID = predictions[i].iid
        itemInfo = itemLUT[itemID]
        rating = contentRecomendation.calculate_rating(weights, fullPredictions[i], itemInfo)
        finalPredictions.append(rating)

    print("Final weights:", weights)

    # Normalize and save final predictions
    dfTargets["Rating"] = finalPredictions
    minRating, maxRating = dfTargets["Rating"].min(), dfTargets["Rating"].max()
    dfTargets["Rating"] = (dfTargets["Rating"] - minRating) * 10 / (maxRating - minRating)

    dfTargets = dfTargets.sort_values(by=["UserId", "Rating"], ascending=[True, False])

    dfResult = dfTargets.drop("Rating", axis=1)
    #dfResult.to_csv(sys.stdout, index=False)

    if args.storeOutput:
        dfResult.to_csv(args.storeOutput, index=False)