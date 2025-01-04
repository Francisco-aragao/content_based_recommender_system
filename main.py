import pandas as pd
import numpy as np
import argparse
from surprise.accuracy import rmse, mae
from surprise import Dataset, Reader, SVD

# TRAIN PARAMS
FACTORS = 100
EPOCHS = 20
LR = 0.005
REG = 0.02
USE_BIAS = True


class RecommenderSVD:
    def __init__(self) -> None:
        self.model = None

    def train(
        self,
        data,
        n_factors=FACTORS,
        n_epochs=EPOCHS,
        lr_all=LR,
        reg_all=REG,
        biased=USE_BIAS,
    ):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            biased=biased,
        )

        self.model.fit(data)

        return self.model

    def test(self, data):
        preds = self.model.test(data)

        return preds, rmse(preds, verbose=False), mae(preds, verbose=False)


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

    train_data = data.build_full_trainset()

    # model
    model = RecommenderSVD()

    model.train(train_data, n_factors=150, n_epochs=25)

    dfTargets = pd.read_csv(args.targets)

    # Convert dfTargets to a list of tuples for prediction
    test_data = list(
        zip(dfTargets["UserId"], dfTargets["ItemId"], [None] * len(dfTargets))
    )

    # Make predictions
    predictions = [
        model.model.predict(uid, iid, r_ui=verdict, verbose=False)
        for (uid, iid, verdict) in test_data
    ]
    fullPredictions = np.array([pred.est for pred in predictions])

    # lookup item info
    ItemLUT = dfContent.set_index("ItemId")[
        ["imdbVotes", "Metascore", "rtRating", "imdbRating", "Awards"]
    ].to_dict(orient="index")

    # The final rating will be a weighted sum of some features
    finalPredictions = []

    for i in range(len(fullPredictions)):
        itemId = predictions[i].iid
        item_info = ItemLUT[itemId]
        rating = (
            0.25
            * fullPredictions[i]
            * 0.7
            * item_info["imdbVotes"]
            * 0.02
            * item_info["Metascore"]
            * 0.02
            * item_info["rtRating"]
            * 0.03
            * item_info["imdbRating"]
            + 6 * item_info["Awards"]
        )

        finalPredictions.append(rating)

    dfTargets["Rating"] = finalPredictions

    # normalize col rating
    minRating = dfTargets["Rating"].min()
    maxRating = dfTargets["Rating"].max()
    dfTargets["Rating"] = ((dfTargets["Rating"] - minRating) * 10) / (
        maxRating - minRating
    )

    # Sort the DataFrame by UserId and then by Rating in descending order
    dfSorted = dfTargets.sort_values(by=["UserId", "Rating"], ascending=[True, False])

    dfSorted.to_csv("target_predictions_sorted.csv", index=False)

    # Drop the Rating column as it's not needed in the final output
    dfResult = dfSorted.drop("Rating", axis=1)

    # Write to a CSV file
    dfResult.to_csv("sorted_items_per_user.csv", index=False)
