# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv('./data/2020-train.csv')

# filter both df's to exclude all the rows with ambiguous pitch calls
# i.e. include only the pitches with either strike or ball called

def filter_df(df):
    strikes = np.array(df['pitch_call'] == 'StrikeCalled')
    balls = np.array(df['pitch_call'] == 'BallCalled')
    filtered = strikes | balls
    return df[filtered]


# some data are skewed
# for ex, pitcher_side has some cells with R, L
# instead of right, left
def fix_skewed_data(df):
    df['pitcher_side'] = np.where(df.pitcher_side == 'R', 'Right', df.pitcher_side)
    df['pitcher_side'] = np.where(df.pitcher_side == 'L', 'Left', df.pitcher_side)
    df['batter_side'] = np.where(df.batter_side == 'R', 'Right', df.batter_side)
    df['batter_side'] = np.where(df.batter_side == 'L', 'Left', df.batter_side)
    return df

# Prepare training dataset
train_df = filter_df(train_df).copy()
train_df = fix_skewed_data(train_df)

X = train_df.copy()
X = X.drop(['pitcher_id', 'batter_id', 'stadium_id', \
                        'umpire_id', 'catcher_id', 'pitch_call', \
                        'pitch_id', 'tilt'], axis = 1)
y = train_df[['pitch_call']]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# convert categorical variables
# Create dummy variables for pitcher_side, batter_side, and pitch_type
# note all the dummies should be 1 category less than their original # of categories
dummies_pitcher_side = pd.get_dummies(X_train[['pitcher_side']])
dummies_pitcher_side = dummies_pitcher_side.iloc[:, 1:]

dummies_batter_side = pd.get_dummies(X_train[['batter_side']])
dummies_batter_side = dummies_batter_side.iloc[:, 1:]

dummies_pitch_type = pd.get_dummies(X_train[['pitch_type']])
dummies_pitch_type = dummies_pitch_type.iloc[:, 0:5]

X_train = X_train.drop(['pitcher_side', 'batter_side', \
                        'pitch_type'], axis = 1)
dummies = pd.concat([dummies_pitcher_side, dummies_batter_side, dummies_pitch_type],
                    axis = 1)
X_train = pd.concat([dummies, X_train], axis = 1)

# convert strikes and balls to zeros/ones
dummies_strikes = pd.get_dummies(y_train[['pitch_call']])
dummies_strikes = dummies_strikes.iloc[:, 1:]
y_train = np.array([x for x in dummies_strikes['pitch_call_StrikeCalled']])



# convert categorical variables
# Create dummy variables for pitcher_side, batter_side, and pitch_type
# note all the dummies should be 1 category less than their original # of categories
dummies_pitcher_side = pd.get_dummies(X_test[['pitcher_side']])
dummies_pitcher_side = dummies_pitcher_side.iloc[:, 1:]

dummies_batter_side = pd.get_dummies(X_test[['batter_side']])
dummies_batter_side = dummies_batter_side.iloc[:, 1:]

dummies_pitch_type = pd.get_dummies(X_test[['pitch_type']])
dummies_pitch_type = dummies_pitch_type.iloc[:, 0:5]

X_test = X_test.drop(['pitcher_side', 'batter_side', \
                        'pitch_type'], axis = 1)
dummies = pd.concat([dummies_pitcher_side, dummies_batter_side, dummies_pitch_type],
                    axis = 1)
X_test = pd.concat([dummies, X_test], axis = 1)

# convert strikes and balls to zeros/ones
dummies_strikes = pd.get_dummies(y_test[['pitch_call']])
dummies_strikes = dummies_strikes.iloc[:, 1:]
y_test = np.array([x for x in dummies_strikes['pitch_call_StrikeCalled']])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

m = len(X_train)
n = len(X_train[0])
rows_to_delete = []

for x in range(0, m):
    nan_found = False
    for y in range(0, n):
        if np.isnan(X_train[x][y]):
            nan_found = True
            break
    if nan_found:
        rows_to_delete.append(x)

X_train = np.delete(X_train, rows_to_delete, axis = 0)
y_train = np.delete(y_train, rows_to_delete, axis = 0)

m = len(X_test)
n = len(X_test[0])
rows_to_delete = []

for x in range(0, m):
    nan_found = False
    for y in range(0, n):
        if np.isnan(X_test[x][y]):
            nan_found = True
            break
    if nan_found:
        rows_to_delete.append(x)

X_test = np.delete(X_test, rows_to_delete, axis = 0)
y_test = np.delete(y_test, rows_to_delete, axis = 0)


# =============================================================================
# # Logistic Regression --> 68.27%
# # Fitting Logistic Regression to the Training set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
#
#
# # KNN --> 82.66%
# # Fitting KNN to the Training set
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
#
#
# # Kernel SVM --> 92.44%
# # Fitting Kernel SVM to the Training set
# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
#
#
# # Naive Bayes --> 82.3%
# # Fitting Naive Bayes to the Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
#
#
# # Decision Tree  --> 89.99%
# # Fitting Decision Tree Classification to the Training set
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# =============================================================================


# Random Forest --> 91.86%
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Test.csv data
# Prepare the df
test_df = pd.read_csv('./data/2020-test.csv')
test_df = fix_skewed_data(test_df)
X_test = test_df.copy()
X_test = X_test.drop(['pitcher_id', 'batter_id', 'stadium_id', \
                        'umpire_id', 'catcher_id', 'is_strike', \
                        'pitch_id', 'tilt'], axis = 1)

# convert categorical variables
# Create dummy variables for pitcher_side, batter_side, and pitch_type
# note all the dummies should be 1 category less than their original # of categories
dummies_pitcher_side = pd.get_dummies(X_test[['pitcher_side']])
dummies_pitcher_side = dummies_pitcher_side.iloc[:, 1:]

dummies_batter_side = pd.get_dummies(X_test[['batter_side']])
dummies_batter_side = dummies_batter_side.iloc[:, 1:]

dummies_pitch_type = pd.get_dummies(X_test[['pitch_type']])
dummies_pitch_type = dummies_pitch_type.iloc[:, 0:5]

X_test = X_test.drop(['pitcher_side', 'batter_side', \
                        'pitch_type'], axis = 1)
dummies = pd.concat([dummies_pitcher_side, dummies_batter_side, dummies_pitch_type],
                    axis = 1)
X_test = pd.concat([dummies, X_test], axis = 1)

# feature scale
X_test = sc_X.transform(X_test)

# remove NaN values
# 2263 rows removed
m = len(X_test)
n = len(X_test[0])
rows_to_delete = []

for x in range(0, m):
    nan_found = False
    for y in range(0, n):
        if np.isnan(X_test[x][y]):
            nan_found = True
            break
    if nan_found:
        rows_to_delete.append(x)

X_test = np.delete(X_test, rows_to_delete, axis = 0)

# predict the results
y_test = classifier.predict(X_test)


# append with balls if a data in a row is NaN
result = []
index = 0
y_index = 0

while y_index < len(y_test):
    if index in rows_to_delete:
        result.append(0)
    else:
        result.append(y_test[y_index])
        y_index += 1
    index += 1

result = pd.DataFrame(result)
result.columns = ['Strikes']
pitch_id = test_df[['pitch_id']]
result = pd.concat([result, pitch_id], axis = 1)
result = pd.DataFrame(result)
result.to_csv('2020-test-predictions.csv',
              index = None,
              encoding = 'utf-8',
              header = True)