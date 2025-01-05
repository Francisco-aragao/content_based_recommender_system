import argparse
import sys

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.accuracy import mae, rmse

# TRAIN PARAMS
FACTORS = 150
EPOCHS = 25
LR = 0.005 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 0.3
WEIGHT_IMDB_VOTES = 0.00001
WEIGHT_METASCORE = 0.3
WEIGHT_RT_RATING = 0.39998
WEIGHT_IMDB = 0.00001
WEIGHT_BIAS_AWARDS = 2

def initParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""Recommender System for the RecSys Challenge II (2024/2).
        Sample usage: python3 main.py ratings.jsonl content.jsonl targets.csv > submission.csv"""
    )

    parser.add_argument(
        "ratings",
        metavar="ratings",
        type=str,
        help="JSONL file with user-item-tstamp ratings to train the model",
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
    return parser


def loadRatings(filename: str):
    dfRatings = pd.read_json(filename, lines=True)

    # Define a reader with the rating scale
    reader = Reader(rating_scale=(min(dfRatings["Rating"]), max(dfRatings["Rating"])))

    # Load the dataset into Surprise
    return (
        Dataset.load_from_df(dfRatings[["UserId", "ItemId", "Rating"]], reader),
        dfRatings,
    )


def loadContent(filename: str):
    dfContent = pd.read_json(filename, lines=True)

    # Getting the Rotten Tomatoes ratings
    allRtRatings = []
    for ratingsList in dfContent["Ratings"]:
        rt = next(
            (
                item["Value"]
                for item in ratingsList
                if item["Source"] == "Rotten Tomatoes"
            ),
            None,
        )

        if rt:
            rt = int(rt[:-1])

        allRtRatings.append(rt)

    dfContent["rtRating"] = allRtRatings

    # Getting useful columns
    dfContentUseful = dfContent[
        ["ItemId", "Metascore", "imdbRating", "imdbVotes", "rtRating", "Awards"]
    ].copy()

    # Updating 'Awards' column
    dfContentUseful["Awards"] = dfContentUseful["Awards"].apply(
        lambda x: 0 if x == "N/A" else 1
    )

    # Replacing string 'N/A' with np.nan and removing number separators
    dfContentUseful = dfContentUseful.replace("N/A", np.nan)
    dfContentUseful["imdbVotes"] = dfContentUseful["imdbVotes"].str.replace(",", "")

    # Converting to numeric data
    dfContentUseful["Metascore"] = dfContentUseful["Metascore"].astype("float32")
    dfContentUseful["imdbRating"] = dfContentUseful["imdbRating"].astype("float32")
    dfContentUseful["imdbVotes"] = dfContentUseful["imdbVotes"].astype("float32")

    # Substitute NaN with mean
    quantiles = dfContentUseful.quantile(0.5, numeric_only=True)
    dfContentUseful = dfContentUseful.fillna(quantiles)

    # Normalizing imdbRating between 0 and 10
    for col in dfContentUseful.columns:
        if col in ["ItemId", "Awards"]:
            continue

        minRating = dfContentUseful[col].min()
        maxRating = dfContentUseful[col].max()

        dfContentUseful[col] = ((dfContentUseful[col] - minRating) * 10) / (
            maxRating - minRating
        )

    return dfContentUseful


if __name__ == "__main__":
    # Get args
    parser = initParser()
    args = parser.parse_args()

    # Load data
    data, dfRatings = loadRatings(args.ratings)
    dfContent = loadContent(args.content)

    trainingData = data.build_full_trainset()

    # Train the model
    model = SVD(
        n_factors=FACTORS,
        n_epochs=EPOCHS,
        lr_all=LR,
        reg_all=REG,
        biased=USE_BIAS,
    )

    model.fit(trainingData)

    dfTargets = pd.read_csv(args.targets)

    # Convert dfTargets to a list of tuples for prediction
    testData = list(
        zip(dfTargets["UserId"], dfTargets["ItemId"], [None] * len(dfTargets))
    )

    # Make predictions
    predictions = [
        model.predict(uid, iid, r_ui=verdict, verbose=False)
        for (uid, iid, verdict) in testData
    ]

    fullPredictions = np.array([pred.est for pred in predictions])

    # lookup item info
    itemLUT = dfContent.set_index("ItemId")[
        ["imdbVotes", "Metascore", "rtRating", "imdbRating", "Awards"]
    ].to_dict(orient="index")

    # The final rating will be a weighted sum of some features
    finalPredictions = []

    for i in range(len(fullPredictions)):
        itemID = predictions[i].iid
        itemInfo = itemLUT[itemID]

        rating = (
            WEIGHT_PREDICTION
            * fullPredictions[i]
            * WEIGHT_IMDB_VOTES
            * itemInfo["imdbVotes"]
            * WEIGHT_METASCORE
            * itemInfo["Metascore"]
            * WEIGHT_RT_RATING
            * itemInfo["rtRating"]
            * WEIGHT_IMDB
            * itemInfo["imdbRating"]
            + WEIGHT_BIAS_AWARDS * itemInfo["Awards"]
        )

        finalPredictions.append(rating)

    dfTargets["Rating"] = finalPredictions

    # Normalize ratings
    minRating = dfTargets["Rating"].min()
    maxRating = dfTargets["Rating"].max()
    dfTargets["Rating"] = ((dfTargets["Rating"] - minRating) * 10) / (
        maxRating - minRating
    )

    # Sort the DataFrame by UserId and then by Rating in descending order
    dfSorted = dfTargets.sort_values(by=["UserId", "Rating"], ascending=[True, False])

    # Drop the Rating column as it's not needed in the final output
    dfResult = dfSorted.drop("Rating", axis=1)

    # Final submission
    dfResult.to_csv(sys.stdout, index=False)
