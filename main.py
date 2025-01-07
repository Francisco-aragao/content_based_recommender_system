from content_recomendation import ContentRecomendation

import argparse
import pandas as pd

# TRAIN PARAMS
FACTORS = 150
EPOCHS = 25
LR = 0.005 
REG = 0.02
USE_BIAS = True

LEARNING_RATE_WEIGHTS = 0.001
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
        "target",
        metavar="target",
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

    model = contentRecomendation.train_SVD_model(data, FACTORS, EPOCHS, LR, REG, USE_BIAS)

    dfTargets = pd.read_csv(args.target)

    rat_predictions, ratings_prediction_nparray = contentRecomendation.generate_predictions(model, dfTargets)

    # Lookup item info
    item_table_info = dfContent.set_index("ItemId")[
        ["imdbVotes", "Metascore", "rtRating", "imdbRating", "Awards"]
    ].to_dict(orient="index")

    weights = contentRecomendation.initialize_weights()

    # Prepare training data
    train_data = [
        (row["UserId"], row["ItemId"], row["Rating"])
        for _, row in dfRatings.iterrows()
    ]

    # Perform gradient descent to learn weights
    weights = contentRecomendation.gradient_descent(weights, train_data, item_table_info, ratings_prediction_nparray, lr=LEARNING_RATE_WEIGHTS, epochs=GRADIENT_EPOCHS)

    # Make final predictions
    finalPredictions = contentRecomendation.get_final_predictions(ratings_prediction_nparray, rat_predictions, item_table_info, weights)

    # print("Final weights:", weights)

    # Normalize and save final predictions
    dfResult = contentRecomendation.get_final_target_df(dfTargets, finalPredictions)

    print(dfResult.to_csv(index=False))