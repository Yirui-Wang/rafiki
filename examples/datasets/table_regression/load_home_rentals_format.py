import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/home_rentals.csv')
    attr = pd.get_dummies(df.iloc[:, 0:df.shape[1]-1])
    df = pd.concat([attr, df.iloc[:, df.shape[1]-1]], axis=1)
    train_length = int(df.shape[0]*2/3)
    df.iloc[:train_length].to_csv('data/home_rentals_dummies_train.csv')
    df.iloc[train_length:].to_csv('data/home_rentals_dummies_test.csv')
