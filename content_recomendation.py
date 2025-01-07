import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

np.random.seed(42)

class ContentRecomendation:

    def __init__(self):
        pass

    def get_final_target_df(self, dfTargets, finalPredictions):
        dfTargets["Rating"] = finalPredictions
        minRating, maxRating = dfTargets["Rating"].min(), dfTargets["Rating"].max()
        dfTargets["Rating"] = (dfTargets["Rating"] - minRating) * 10 / (maxRating - minRating)

        # sort values to create ranking
        dfTargets = dfTargets.sort_values(by=["UserId", "Rating"], ascending=[True, False])

        dfResult = dfTargets.drop("Rating", axis=1)
        
        return dfResult
    
    def get_final_predictions(self, ratings_prediction_nparray, rat_predictions, item_table_info, weights):
        finalPredictions = []
        for i in range(len(ratings_prediction_nparray)):
            itemID = rat_predictions[i].iid
            itemInfo = item_table_info[itemID]
            rating = self.calculate_rating(weights, ratings_prediction_nparray[i], itemInfo)
            finalPredictions.append(rating)

        return finalPredictions

    def train_SVD_model(self, data, factors: int, epochs: int, lr: float, reg: float, use_bias: bool):
        trainingData = data.build_full_trainset()

        # Train the model
        model = SVD(
            n_factors=factors,
            n_epochs=epochs,
            lr_all=lr,
            reg_all=reg,
            biased=use_bias,
        )
        model.fit(trainingData)

        return model

    def generate_predictions(self, model, dfTargets):
        
        testData = list(
            zip(dfTargets["UserId"], dfTargets["ItemId"], [None] * len(dfTargets))
        )
        ratings_prediction = [
            model.predict(uid, iid, r_ui=None, verbose=False) for (uid, iid, _) in testData
        ]

        ratings_prediction_nparray = np.array([pred.est for pred in ratings_prediction])

        return ratings_prediction, ratings_prediction_nparray

    def loadRatings(self, filename: str):
        dfRatings = pd.read_json(filename, lines=True)

        # Define a reader with the rating scale
        reader = Reader(rating_scale=(min(dfRatings["Rating"]), max(dfRatings["Rating"])))

        # Load the dataset into Surprise
        return (
            Dataset.load_from_df(dfRatings[["UserId", "ItemId", "Rating"]], reader),
            dfRatings,
        )


    def loadContent(self, filename: str):
        dfContent = pd.read_json(filename, lines=True)

        # Extracting Rotten Tomatoes ratings
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

        # Substitute NaN with median
        quantiles = dfContentUseful.quantile(0.5, numeric_only=True)
        dfContentUseful = dfContentUseful.fillna(quantiles)

        # Normalizing ratings between 0 and 10
        for col in dfContentUseful.columns:
            if col in ["ItemId", "Awards"]:
                continue

            minRating = dfContentUseful[col].min()
            maxRating = dfContentUseful[col].max()

            dfContentUseful[col] = ((dfContentUseful[col] - minRating) * 10) / (
                maxRating - minRating
            )

        return dfContentUseful


    def calculate_rating(self, weights, prediction, item_info):
        return (
            weights["prediction"] * prediction
            + weights["imdb_votes"] * item_info["imdbVotes"]
            + weights["metascore"] * item_info["Metascore"]
            + weights["rt_rating"] * item_info["rtRating"]
            + weights["imdb"] * item_info["imdbRating"]
            + weights["bias_awards"] * item_info["Awards"]
        )

    def gradient_descent(self, weights, train_data, item_lut, full_predictions, lr=0.001, epochs=100):
        """Perform gradient descent to learn weights and calculate RMSE."""
        for _ in range(epochs):
            gradients = {k: 0.0 for k in weights.keys()}
            errors = []

            for i in range(len(full_predictions)):
                item_id = train_data[i][1]
                actual_rating = train_data[i][2] 

                item_info = item_lut[item_id]
                predicted_rating = self.calculate_rating(weights, full_predictions[i], item_info)

                # compute error
                error = predicted_rating - actual_rating
                errors.append(error)

                # find gradients
                gradients["prediction"] += error * full_predictions[i]
                gradients["imdb_votes"] += error * item_info["imdbVotes"]
                gradients["metascore"] += error * item_info["Metascore"]
                gradients["rt_rating"] += error * item_info["rtRating"]
                gradients["imdb"] += error * item_info["imdbRating"]
                gradients["bias_awards"] += error * item_info["Awards"]

            # Update weights 
            for k in weights.keys():
                weights[k] -= lr * gradients[k] / len(full_predictions)  # Normalize by batch size

            # calculate RMSE errors
            #rmse = np.sqrt(np.mean(np.array(errors) ** 2))
            #print(f"Epoch {epoch + 1}/{epochs}, RMSE: {rmse:.4f}")

        return weights

    def initialize_weights(self, bias_key="bias_awards"):
        # the weights are not totally random. They are initialized with some intuition based on Kaggle experimentation
        weights = {
            "prediction": 0.2,
            "imdb_votes": 0.7,
            "metascore": 0.02,
            "rt_rating": 0.03,
            "imdb": 0.05,
        }
        
        # Assign a separate random value to the bias key
        weights[bias_key] = np.random.randint(1, 5)  # Random value between 0 and 5

        return weights