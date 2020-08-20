from .head import *
from lightgbm import LGBMClassifier
import lightgbm as lgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import PredefinedSplit
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot
import keras
from keras import layers
import joblib



class Model:
    """Basic class for feature selection, model building and turning."""

    ## Feature selection
    def feature_importance_lgb(self, y_train, X_train, nfolds=5, nrepeats=2):
        """ By using lightGBM to get feature importance

        Args:
           y_train: series: data response
           X_train: Dataframe df:
           nfolds: int: fold number
           nrepeats: int: number of repeats for CV

        return:
           a dataframe with feature name and features' importance. The features are in descending order based on their importance
        """
        folds = sklearn.model_selection.RepeatedKFold(n_splits=nfolds, n_repeats=nrepeats)
        feature_importance_df = pd.DataFrame()
        model0 = LGBMClassifier(is_unbalance=True)
        for fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold n°{}".format(fold))
            model0.fit(X_train.iloc[trn_idx], y_train.iloc[trn_idx],
                       eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])], early_stopping_rounds=150)
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = X_train.columns
            fold_importance_df["importance"] = model0.booster_.feature_importance()
            fold_importance_df["fold"] = fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        lgb.plot_importance(model0.booster_).plot()
        plt.title("Feature Importance")
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()
        all_features = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)
        all_features.reset_index(inplace=True)
        return all_features

    def feature_correlation_lgb(self, X_train, important_features, importance_threshold, correlation_threshold):
        """ Select features which have high importance score and low correlation. For the feature which are highly correlated,
        we pick the one with highest importance score from correlated features sets. The correlation matrix idea is from :https://www.kaggle.com/juliaflower/feature-selection-lgbm-with-python

       Args:
          X_train: Dataframe df:
          important_features: Dataframe df: The table which is generated from the function - feature_importance_lgb
          importance_threshold: threshold on feature importance. It serves as a lower bound on feature importance selection
          correlation_threshold: threshold on correlation between features. It serves as a upper bound on feature correlation selection
       return:
            a list of features which have high importance score and low correlation

       """
        features = important_features[important_features['importance'] > importance_threshold]
        important_features = list(features['feature'])
        print(important_features)
        df = X_train[important_features]
        corr_matrix = df.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than 0.95
        high_cor = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        print("# of Correlated features is {} and they are {}".format(len(high_cor), high_cor))
        features = [i for i in important_features if i not in high_cor]
        print("# of Uncorrelated features is {} and they are {}".format(len(features), features))
        return features


    ## LightGBM Fitting
    # def fit_original_lgb_1(self, X_train, y_train, X_test, y_test):
    #     model0 = LGBMClassifier(is_unbalance=True, reg_lambda=1)
    #     model0.fit(X_train, y_train)
    #     joblib.dump(model0, 'lgb_model_original.pkl')
    #     result = model0.predict(X_test)
    #     f1 = sklearn.metrics.f1_score(y_test, result)
    #     return f1

    def fit_lgb(self, X_train, y_train, X_val, y_val, X_test, y_test, **param):
        """ using turned parameters to fit training dataset, and save the fitted model to a txt file. Also, it return f1
        score on test set.

        Args:
           X_train: Dataframe df: train set
           y_train: series: train set response
           X_val: Dataframe df: validation set
           y_val: series: validation set response
           X_test: Dataframe df: test set
           y_test: series: test set response
           **param: LightGBM parameters selected from function - turining_lgb()
        return:
             f1_score for test set

        """
        model0 = LGBMClassifier(is_unbalance=True, reg_lambda=1)
        model0.set_params(**param)
        model0.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=150)
        fold_pred = model0.predict(X_test, num_iteration=model0.best_iteration_)
        fold_pred_prob = model0.predict_proba(X_test, num_iteration=model0.best_iteration_)
        model_probs = fold_pred_prob[:, 1]

        joblib.dump(model0, 'lgb_model.pkl')
        f1 = sklearn.metrics.f1_score(y_test, fold_pred)
        print("lgb turning model - f1_score:{}".format(f1))
        lgb.plot_importance(model0.booster_).plot()
        plt.title("Feature Importance for selected features in the final LGB model")
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()

        # calculate the precision-recall auc
        precision, recall, _ = precision_recall_curve(y_test, model_probs)
        auc_score = auc(recall, precision)
        print('AUC: %.3f' % auc_score)
        # plot precision-recall curves
        self.plot_pr_curve(y_test, model_probs)

        # ROC
        sklearn.metrics.plot_roc_curve(model0, X_test, y_test)
        plt.title("ROC for NN + lightGBM")
        plt.show()

        return f1

    def turning_lgb(self, X_train, y_train, X_val, y_val):
        """ Applying Randomized search on turning lgb's parameters on validation dataset

        Args:
           X_train: Dataframe df: train set
           y_train: series: train set response
           X_val: Dataframe df: validation set
           y_val: series: validation set response
        return:
             the turned parameters for lgb

        """
        param_test = {
            'min_child_samples': sp_randint(10, 100),
            'subsample': sp_uniform(loc=0.2, scale=0.8),
            'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1],
            'colsample_bytree': sp_uniform(loc=0.4, scale=0.6)}
        # This parameter defines the number of HP points to be tested
        n_HP_points_to_test = 300
        clf = lgb.LGBMClassifier(is_unbalance=True)
        gs = RandomizedSearchCV(
            estimator=clf, param_distributions=param_test,
            n_iter=n_HP_points_to_test,
            scoring='f1',
            cv=3,
            refit=True,
            verbose=False)
        fit_params = {"early_stopping_rounds": 30,
                      "eval_metric": ['logloss'],
                      "eval_set": [(X_val, y_val)],
                      'eval_names': ['valid']}
        gs.fit(X_train, y_train, **fit_params)
        print('Best f1_score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
        return (gs.best_params_)

    ## Neural Network
    def parameter_ANN(self, X_train, y_train, X_val, y_val, neuron_layer_1):
        """ Applying Gridsearch on turning neural net's parameters on validation dataset. It will write the final model
        in a txt file, whose hidden layers will be used to extract new features.

        Args:
           X_train: Dataframe df: train set
           y_train: series: train set response
           X_val: Dataframe df: validation set
           y_val: series: validation set response
           neuron_layer_1: int: the number of neurons of the input layers. This number is decided by the number of selected features

        """
        X_train = X_train.drop(X_train.select_dtypes(include='category'), axis=1)
        X_val = X_val.drop(X_val.select_dtypes(include='category'), axis=1)

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        ## standaderization
        scalar = StandardScaler()
        scalar.fit(X_train)
        transformed_X_train = scalar.transform(X_train)
        transformed_X_val = scalar.transform(X_val)

        ## parameter choice
        neuron = [16, 32, 64]
        dropout_rate = [0.2, 0.4]
        activations = ['softmax', 'relu', 'tanh', 'sigmoid']

        ## baseline model
        def create_baseline(dropout_rate=0.0, neurons=10, activation1='relu', activation2='relu'):
            # create model
            model = Sequential(
                [
                    layers.Dense(neuron_layer_1, activation=activation1, name="layer1"),
                    layers.Dropout(dropout_rate),
                    layers.Dense(neurons, activation=activation2, name="layer2"),
                    layers.Dense(1, activation='sigmoid', name="layer3")
                ]
            )
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        X_val_train = np.vstack((transformed_X_val, transformed_X_train))
        y_val_train = np.vstack((y_val[:, np.newaxis], y_train[:, np.newaxis]))
        split_index = [0] * transformed_X_val.shape[0] + [-1] * transformed_X_train.shape[0]
        model = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=25, verbose=0)
        param_grid = dict(activation1=activations, activation2=activations, neurons=neuron, dropout_rate=dropout_rate)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=PredefinedSplit(test_fold=split_index))
        grid_result = grid.fit(X_val_train, y_val_train)

        ## summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        ## save weights and bias
        m = grid_result.best_estimator_
        m.model.save("nn_model.h5")
        print(m.model.summary())
        return 0

    def feature_generation_NN(self, X_train, X_val, X_test, y_test, evaluation=None):
        """ Get hidden layers from the saved NN model, and use it to generate new features on train, test, validation and
        evaluation dataset if it is applicable

         Args:
            X_train: Dataframe df: train set
            X_val: Dataframe df: validation set
            X_test: Dataframe df: test set
            y_test: series: validation set response
            evaluation: DataFrame df: evaluation dataset
        return:
             the train,test,validation and evaluation datasets with new added features
         """
        # import the turned model
        model = keras.models.load_model("nn_model.h5")
        # dot_img_file = '../graph/model_1.png'
        # keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)
        index_train = X_train.index
        index_test = X_test.index
        index_val = X_val.index

        X_train = X_train.drop(X_train.select_dtypes(include='category'), axis=1)
        X_test = X_test.drop(X_test.select_dtypes(include='category'), axis=1)
        X_val = X_val.drop(X_val.select_dtypes(include='category'), axis=1)

        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        ## standaderization
        scalar = StandardScaler()
        scalar.fit(X_train)
        transformed_X_train = scalar.transform(X_train)
        transformed_X_val = scalar.transform(X_val)
        transformed_X_test = scalar.transform(X_test)

        ## feature extraction
        extractor = keras.Model(inputs=model.inputs,
                                outputs=model.layers[2].output)
        features_train = extractor.predict(transformed_X_train)
        features_test = extractor.predict(transformed_X_test)
        features_val = extractor.predict(transformed_X_val)

        col = ["val" + str(i) for i in range(model.layers[2].output_shape[1])]
        feature_train = pd.DataFrame(data=features_train, index=index_train, columns=col)
        feature_test = pd.DataFrame(data=features_test, index=index_test, columns=col)
        feature_val = pd.DataFrame(data=features_val, index=index_val, columns=col)

        ## evaluate model
        predict = model.predict_classes(transformed_X_test)
        f1 = f1_score(y_test, predict)
        print("f1 score for neural net: {}".format(f1))

        if evaluation is not None:
            index_evaluation = evaluation.index
            evaluation = evaluation.drop(evaluation.select_dtypes(include='category'), axis=1)
            evaluation = np.asarray(evaluation)
            transformed_evaluation = scalar.transform(evaluation)
            features_evaluation = extractor.predict(transformed_evaluation)
            feature_evaluation = pd.DataFrame(data=features_evaluation, index=index_evaluation, columns=col)
            return feature_test, feature_train, feature_val, feature_evaluation, f1
        return feature_test, feature_train, feature_val, f1

    def plot_pr_curve(self,test_y, model_probs):
        """ Drawing the Precision-Recall Curves. This code is borrowed from:https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/#:~:text=The%20Precision%2DRecall%20AUC%20is,a%20model%20with%20perfect%20skill.

        Args:
           test_y: series: test set response
           model_probs: The probability that the model decide the instance to be in class 1
        return:
             The graph of Precision-Recall Curves

        """
        # calculate the no skill line as the proportion of the positive class
        no_skill = len(test_y[test_y == 1]) / len(test_y)
        # plot the no skill precision-recall curve
        pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        # plot model precision-recall curve
        precision, recall, _ = precision_recall_curve(test_y, model_probs)
        pyplot.plot(recall, precision, marker='.', label='NN+lightGBM')
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.title("Precision-Recall Curves")
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()
