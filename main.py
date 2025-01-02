# execution: python3 main.py ratings.csv targets.csv

import argparse # default python library

import pandas as pd
import json

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

    args = parser.parse_args()

    if not (args.ratings and args.targets and args.content):
        raise Exception

    with open(args.ratings, 'r', encoding='utf-8') as file:
        ratings = [json.loads(line.strip()) for line in file]
    
    with open(args.content, 'r', encoding='utf-8') as file:
        content = [json.loads(line.strip()) for line in file]


    targets = pd.read_csv(args.targets)

    store_output = args.storeOutput if args.storeOutput else None

    return ratings, content, targets, store_output


def store_print_output(key: str, targets: pd.DataFrame, recomendations: list):
    """
        Store output in a csv file or just print in the console based on the key received
    """

    # to store the results, just copy the target dataset and add a new column with the recomendations
    df = targets.copy()
    df["Rating"] = recomendations

    if key:

        df.to_csv('submission.csv', index=False)

    print("UserId:ItemId,Rating")
    for _, row in df.iterrows():
        print(row["UserId:ItemId"] + "," + str(row["Rating"]))



if __name__ == "__main__":

    # receive parameters
    ratings, content, targets, output_flag = receive_args()
