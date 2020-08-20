
if __name__ == '__main__':

    from src.competition.Model import Model
    from src.competition.CleanData import CleanData
    from src.competition.head import *
    import joblib
    import pickle
    data = CleanData()
    m = Model()
    challenge,evaluation = data.read_data()
    dataset_id = evaluation["scenario"]
    evaluation = evaluation.iloc[:,0:-1]
    X_train,y_train,X_test,y_test,X_val,y_val,evaluation = data.data_split(challenge,chang_category=True, evaluation = evaluation)


    ## feature selection
    result = m.feature_importance_lgb(y_train,X_train)
    selected_feature=m.feature_correlation_lgb(X_train,result,50,0.80)
    with open("first_run_selection.txt", "wb") as fp:   #Pickling
       pickle.dump(selected_feature, fp)

    ## NN
    with open("first_run_selection.txt", "rb") as fp:   # Unpickling
        selected_feature_run_1 = pickle.load(fp)
    X_train = X_train[selected_feature_run_1]
    X_val=X_val[selected_feature_run_1]
    X_test=X_test[selected_feature_run_1]
    evaluation=evaluation[selected_feature_run_1]
    num_features =len(selected_feature_run_1)

    ## NN feature_generation
    m.parameter_ANN(X_train,y_train,X_val,y_val,num_features)
    feature_test,feature_train,feature_val,feature_evaluation,f1 = m.feature_generation_NN(X_train, X_val, X_test, y_test, evaluation=evaluation)
    new_X_test = pd.concat([X_test,feature_test], axis=1, join='inner')
    new_X_val = pd.concat([X_val,feature_val], axis=1, join='inner')
    new_X_train = pd.concat([X_train,feature_train], axis=1, join='inner')
    modified_evaluation = pd.concat([evaluation,feature_evaluation], axis=1, join='inner')

    ##LightGBM Turning
    params = m.turning_lgb(new_X_train,y_train,new_X_val,y_val)
    f1_lgb_turning=m.fit_lgb(new_X_train,y_train,new_X_val,y_val,new_X_test,y_test,**params)


    ##Testing
    gbm_pickle_turning = joblib.load('lgb_model.pkl')
    y_pred_turning = pd.Series(gbm_pickle_turning.predict(modified_evaluation)).to_frame()

    ## Prediction
    final_result = pd.concat([dataset_id,y_pred_turning],axis =1)
    final_result = final_result.rename(columns={"scenario":"dataset_id",0: "prediction_score"}).set_index("dataset_id")
    final_result.to_csv('../data/deliverable_1_result.csv')

