from sklearn import ensemble
MODELS = {
'randomforest':ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1, verbose=2),
'gradientboost':ensemble.GradientBoostingClassifier(n_estimators=200,verbose=2)
}