from data.loader import load_movielens_100k, dataset_to_user_item_lists

def main():
    path = "data/raw/ml-100k/u.data"

    # load dataset
    df = load_movielens_100k(path)
    print("Loaded dataset:")
    print(df.head())

    # convert to user -> items list
    users = dataset_to_user_item_lists(df)
    print("Number of users:", len(users))

if __name__ == "__main__":
    main()
