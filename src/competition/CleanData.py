from .head import *
import tabulate
from sklearn.model_selection import train_test_split


class CleanData:
    """Basic class for data importing, exploration, cleaning and test-train-validation splitting."""

    def read_data(self):
        """imports and change the XC column to category type."""
        challenge = pd.read_excel('../data/0173eeb640e7-Challenge+Data+Set+-+Campus+Analytics+2020.xlsx')
        evaluation = pd.read_excel('../data/d59675225279-Evaluation+Data+Set+-+Campus+Analytics+2020.xlsx')
        challenge[['XC']] = challenge[['XC']].astype('category')
        evaluation[['XC']] = evaluation[['XC']].astype('category')
        return challenge,evaluation

    def exploration(self,data):
        """explore the data in term of basic statistics, missing variable, and data type
        Args:
           data:  DataFrame df: challenge/evaluation
        """
        size = data.shape
        missing_num = data.isnull().sum()
        print("Dataset Statistics")
        print(tabulate.tabulate(data.describe(include='all'), headers="keys"))
        print("Variable Types")
        print(data.dtypes)
        print("Dataset size")
        print(size)
        print("Distribution on column 'XC'")
        print(data.groupby('XC').count()/data.shape[0])
        print("Missing Value Checking")
        print(missing_num)

    def data_split(self,data,chang_category=True, evaluation = None):
        """ splitting the challenge data into train, test and validation. Changing the categorical variable in the
            challenge dataset and evaluation dataset if user want

        Args:
           data: DataFrame df: challenge dataset
           chang_category: boolean: change the category variable to numeric variable  by replacing the level with the average
           on the y under the associated level. The average on the y will be calculated from the training dataset, and applied
            to the validation, testing and evaluation
           evaluation: DataFrame df: evaluation dataset

        return:
           modified train, test, validation and evaluation
        """

        X = data.iloc[:,0:-1]
        y = data.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=2/9, random_state=1)
        if chang_category == True:
            type, value = self.categorical_checking(X_train, y_train)
            X_test = self.categorical_changing(X_test, type, value)
            X_val  = self.categorical_changing(X_val,type,value)
            X_train = self.categorical_changing(X_train,type,value)
            evaluation = self.categorical_changing(evaluation,type,value)
            y_train = y_train.astype("category")
            y_test = y_test.astype("category")
            y_val = y_val.astype("category")
            return X_train,y_train,X_test,y_test,X_val,y_val,evaluation
        y_train = y_train.astype("category")
        y_test = y_test.astype("category")
        y_val = y_val.astype("category")
        return X_train, y_train, X_test, y_test, X_val, y_val

    def categorical_checking(self, data_X,data_y):
        """ Find out the category variable, and find the mean on the y under the associated level

        Args:
           data_X: DataFrame df: training dataset'X
           data_y: Series: training dataset Y

        return:
           A,B,C,D,E and their associated averaged y
        """
        data = pd.concat([data_X, data_y], axis=1)
        XC = data.select_dtypes(include='category')
        type = np.unique(XC.values.tolist())
        dist = []
        for char in type:
           dist.append(np.mean(data.query('XC=="%s"'%(char)).iloc[:,-1]))
        return type,dist

    def categorical_changing(self, data, type, dist):
        """ Replace the category variable with its associated averaged y

        Args:
           data: DataFrame df
           type: categorical variables' levels
           dist: the averaged y under the associated level

        return:
           modified dataset
        """
        data[['XC']] = data[['XC']].replace(type,dist)
        return data




##if you want to run the following, please change line 12's address to "../../data/challenge.xlsx"
## and change line 13's address to "../../data/evaluation.xlsx"
if __name__ == "__main__":
    data = CleanData()
    challenge,evaluation = data.read_data()
    data.exploration(challenge)
    data.exploration(evaluation)
    type,dist = data.categorical_checking(challenge.iloc[:,0:-1],challenge.iloc[:,-1])
    print(type)
    print(dist)


