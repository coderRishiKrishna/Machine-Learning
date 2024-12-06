import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        pass
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def load_data(path):
    data = pd.read_csv(path)
    return data

def merge_test_train_data(train,test):
    train["source"] = "train"
    test["source"] = "test"
    data = pd.concat([train,test],ignore_index=True,sort = True)
    return data

def data_properties(data):
    print('\n',"Shape of the data : ",data.shape)
    print("\n","---------------------------------------------------","\n")    
    print("Index values of the data: ",data.index,"\n","Column names of the dataa :",data.columns)
    print("\n","---------------------------------------------------","\n")
    print("Number of null values in the data : ","\n",data.apply(lambda x: sum(x.isnull())))
    print("\n","---------------------------------------------------","\n")
    print("Unique values in the data: ","\n",data.apply(lambda x: len(x.unique())))
    print("\n","---------------------------------------------------","\n")
    category_columns = [column for column in data.dtypes.index if data.dtypes[column] == 'object']
    req_cat_col = [column for column in category_columns if column not in ['Item_Identifier','Outlet_Identifier','source']]
    print("Frequency count for Categorical values is as :","\n","-------------------- *****************--------------","\n")
    for col in req_cat_col:
        print("The Frequency Count for the column ",col," is :")
        print(data[col].value_counts())
        print("\n","---------------------------------------------------","\n")
   
    
def clean_data(data):
    # Step 1 - Fill in the Missing values
    # the missing values in "Outlet Weight" is filled by mean of the column and in "Outlet_size" by mode of the column
    imp = DataFrameImputer()
    imp.fit(data)
    data = imp.transform(data)
    #step 2 - correct the labelling of the "Item_FAT_CONTENT " column . 
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
    #step 3 - crerate a new feature denoting the working time period of the Outlet
    data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year'] # used 2013 since data collected till 2013
    #step 4 - OneHot Encoding for the categorical values 
    data = pd.get_dummies(data,columns=["Outlet_Years","Outlet_Type","Outlet_Size","Outlet_Location_Type","Outlet_Identifier","Item_Type","Item_Fat_Content"])
    # print(data.dtypes)
    # step 5 - Remove the Outlet_Establishment_Year from the data 
    data.drop(["Outlet_Establishment_Year"],axis = 1,inplace = True)
    # print(data.dtypes)

    #step 6 - split the data into test and train dataset
    train = data.loc[data["source"]=="train"]
    test = data.loc[data["source"]=="test"]
    # print(test.shape,train.shape)
    # step 7 - remove the unnecessary columns from the dataset i.e of "source" from train and of "train","Item_Outlet_Sales"(to predict) from the test
    train.drop('source',axis = 1,inplace = True,errors = 'ignore')
    test.drop(['Item_Outlet_Sales','source'],axis = 1,inplace = True,errors = 'ignore')
    # print(test.shape,train.shape)
    return test,train 
    
def evaluate_model(model,train,test,text,learning_rate = None):
    features = [x for x in train.columns if x not in ['Item_Identifier','Item_Outlet_Sales']]
    target = 'Item_Outlet_Sales'
    model.fit(train[features],train[target])
    train_pred = model.predict(train[features])
    RMSE = np.sqrt(metrics.mean_squared_error(train[target].values, train_pred))
    cv_score = cross_validate(model,train[features], train[target], scoring='neg_mean_squared_error',cv=3)
    print("Report for the model : ",text,"\n")
    print("RMSE = ",RMSE)
    print("CV Score : Mean : %.4g | Std : %.4g | Min : %.4g | Max : %.4g" % (np.mean(cv_score['test_score']),np.std(cv_score['test_score']),np.min(cv_score['test_score']),np.max(cv_score['test_score'])))
    print("Predicting on the Test Data : ","\n")
    print("\n","---------------------------------------------------","\n")
    test[target] = model.predict(test[features])
    submission_col = ['Item_Identifier',target]
    submission = pd.DataFrame({ x: test[x] for x in submission_col})
    submission.to_csv(text+"result.csv", index=False)
    coef1 = pd.Series(model.coef_, features).sort_values()


def evaluate_linear_classifiers(linear_models,text):
    for i in range(len(linear_models)):
        model = linear_models[i](normalize = True)
        evaluate_model(model,train,test,text[i])
def evaluate_random_forest(train,test,model):
    features = [x for x in train.columns if x not in ['Item_Identifier','Item_Outlet_Sales']]
    target = 'Item_Outlet_Sales'
    submission_col = ['Item_Identifier',target]
    model.fit(train[features],train[target])
    train_pred = model.predict(train[features])
    RMSE = np.sqrt(metrics.mean_squared_error(train[target].values, train_pred))
    cv_score = cross_validate(model,train[features], train[target], scoring='neg_mean_squared_error',cv=3)
    print("Report for the model : ","Random_Forest","\n")
    print("RMSE = ",RMSE)
    print("CV Score : Mean : %.4g | Std : %.4g | Min : %.4g | Max : %.4g" % (np.mean(cv_score['test_score']),np.std(cv_score['test_score']),np.min(cv_score['test_score']),np.max(cv_score['test_score'])))
    print("Predicting on the Test Data : ","\n")
    print("\n","---------------------------------------------------","\n")
    # coef5 = pd.Series(model.feature_importances_, features).sort_values(ascending=False)
    test[target] = model.predict(test[features])
    submission_col = ['Item_Identifier',target]
    submission = pd.DataFrame({ x: test[x] for x in submission_col})
    submission.to_csv("Random_Forest.csv", index=False)
    
    # print(coef5)
    
    

if __name__ == "__main__":
    train,test = load_data("Train.csv"),load_data("Test.csv")
    data = merge_test_train_data(train,test) 
    test,train = clean_data(data)
    
    linear_models = [LinearRegression, Ridge, Lasso]
    text = ["LinearRegression", "Ridge", "Lasso"]
    evaluate_linear_classifiers(linear_models,text)

    model = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
    evaluate_random_forest(train,test,model)
    
 
    
    
