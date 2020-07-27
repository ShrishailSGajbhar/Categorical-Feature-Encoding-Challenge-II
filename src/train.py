import os
import pandas as pd
from sklearn import ensemble
import category_encoders as ce 
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target
    yvalid = valid_df.target

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    cols_to_enc = train_df.columns.tolist()

    target_encoders = {}
    for c in train_df.columns:
        if c in cols_to_enc:
            print(c)
            tge = ce.TargetEncoder(cols=[c])
            tge.fit(pd.concat([train_df.loc[:,c],valid_df.loc[:,c]],axis=0), pd.concat([ytrain,yvalid],axis=0))
            train_df.loc[:,c] = tge.transform(train_df.loc[:,c])
            valid_df.loc[:,c] = tge.transform(valid_df.loc[:,c])
            target_encoders[c] = tge
    
    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(target_encoders, f"models/{MODEL}_{FOLD}_target_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")