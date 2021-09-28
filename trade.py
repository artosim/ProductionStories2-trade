import pandas as pd
import numpy as np

def prepare_data(data):
    X_train = []
    ses_id = 0
    new_ses = [0.0] * 60
    for index, row in data.iterrows():
        if row['session_id'] != ses_id:
            X_train.append(new_ses)
            ses_id = row['session_id']
            new_ses = [0.0] * 60
        new_ses[int(row['time'][3:-3])] = row['norm_price']
    X_train.append(new_ses)
    X_train = X_train[1:]

    for ses in range(len(X_train)):
        x = 0.0
        for i in range(60):
            if X_train[ses][i] == 0:
                X_train[ses][i] = x
            else:
                x = X_train[ses][i]

    X_train = np.array(X_train)
    X_train = np.expand_dims(X_train, axis=2)
    return X_train
