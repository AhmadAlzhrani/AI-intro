import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)
   

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    file = pd.read_csv(filename,header=0)

    month = {'Jan':0, 'Feb':1, 'Mar':2, 'Apr':3, 'May':4, 'June':5, "Jul":6, 'Aug':6, 'Sep':8, 'Oct':9, 'Nov':10, 'Dec':11}

    file['Month'] = file['Month'].map(month)

    file['VisitorType'] = file['VisitorType'].map(lambda x : 1 if x=='Returning_Visitor' else 0 )

    file['Weekend'] = file['Weekend'].map(lambda x : 1 if x==True else 0 )

    file['Revenue'] = file['Revenue'].map(lambda x : 1 if x==True else 0 )

    nums = ['Administrative','Informational','ProductRelated','Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']

    doubles = ['Administrative_Duration','Informational_Duration','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay']

    for v in nums:
        if file[v].dtype != 'int64':
            file = file.astype({v: 'int64'})
       
    for v in doubles:
        if file[v].dtype != 'float64':
            file = file.astype({v:'float64'})

    evi = file.iloc[:,:-1].values.tolist()
    label = file.iloc[:,-1].values.tolist()

    if len(evi) != len(label):
        print("number of evidences != number of labels ")
    else:
        print("there is %.0i labels and evidences in the data"%len(evi))
    
    return(evi,label)

    raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    
    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(evidence,labels)

    return knn

    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    pos = labels.count(1)
    neg = labels.count(0)
    sensCount = 0
    specCount = 0

    for L , P in zip(labels, predictions):
        if L ==1 :
            if L == P:
                sensCount+=1
        else:
            if L == P:
                specCount+=1
    
    return (sensCount/pos, specCount/neg)

    raise NotImplementedError


if __name__ == "__main__":
    main()
