#Import libraries
import pandas as pd
import numpy
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


#Initialize constants
NTRAIN = 150
SEED = 545
NPARRAY = 0     #1 if the data should be an array; 0 if pandas df
NSTATE = 1
DEPTH = 5
NLEAF = 2
NTREES = 10
BAG_PROP = .6

#set the random seed
numpy.random.seed(SEED)


#Import data
cars = pd.read_csv('car.csv')

COLNAMES = list(cars.columns.values)    #Save column names to a list

#Reshape data into a numeric array
#Replace car text data cells as follows:
    #buying and maint variables
        #vhigh -> 4
        #high -> 3
        #med -> 2
        #low -> 1
cars = cars.replace('vhigh',4)
cars = cars.replace('high',3)
cars = cars.replace('med',2)
cars = cars.replace('low',1)
    #numeric variables
cars = cars.replace('4',4)
cars = cars.replace('3',3)
cars = cars.replace('2',2)
cars = cars.replace('5more',5)
cars = cars.replace('more',5)
    #lug_boot and safety variable
cars = cars.replace('big',3)
cars = cars.replace('small',1)
    #y variable
cars = cars.replace('vgood',4)
cars = cars.replace('good',3)
cars = cars.replace('acc',2)
cars = cars.replace('unacc',1)


#If NPARRAY = 1, rewrite the pandas dataframe as a numpy array so that we can
    # treat it the same way we've been treating other datasets throughout this
    # semester
if NPARRAY == 1:
    cars = cars.values

#Split into training and test data
    #The original dataset is sorted by the value of the first column, so we
    # need to shuffle the order of the rows of the array or else our training
    # data will only contain specific values of one variable
numpy.random.shuffle(cars)
cars_train = cars[0:NTRAIN]
cars_test = cars[NTRAIN:len(cars)]

def generate_tree(training_data):
    global NSTATE
    global NLEAF
    global COLNAMES
    tree = DecisionTreeClassifier(random_state=NSTATE,min_samples_leaf=NLEAF,  \
                                     splitter = 'random', max_features = 'auto')
    tree.fit(training_data[COLNAMES], training_data["y"])
    return tree


def predict(data):
    global COLNAMES
    prediction = tree1.predict(data[COLNAMES])



def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column
    counts = numpy.bincount(column)
    # Divide by the total column length to get a probability
    probabilities = counts / float(len(column))
    
    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    
    return -entropy
  
def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a data set, column to split on, and target
    """
    # Calculate the original entropy
    original_entropy = calc_entropy(data[target_name])
    
    # Find the median of the column we're splitting
    column = data[split_name]
    median = numpy.median(column)
    
    # Make two subsets of the data, based on the median
    left_split = data[column <= median]
    right_split = data[column > median]
    
    # Loop through the splits and calculate the subset entropies
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0]) 
        to_subtract += prob * calc_entropy(subset[target_name])
    
    # Return information gain
    return original_entropy - to_subtract
 
def find_best_column(data, target_name, columns):
    # Fill in the logic here to automatically find the column in columns to split on
    # data is a dataframe
    # target_name is the name of the target variable
    # columns is a list of potential columns to split on
    information_gains = []
    for col in columns:
        information_gains.append(calc_information_gain(data, col, target_name))
    best_index = information_gains.index(max(information_gains))
    return columns[best_index]

def sample_with_replacement(data):
    """
    Create a sample with replacement for the bootstrap aggregation process
    """
    sample = []
    while len(sample) < len(cars_train):
        sample.append(cars_train[numpy.random.randint(0,len(cars_train)),:])
    return sample

def check_prediction(y, predicted_y):
    """
    Check the accuracy of a predicted y column against the real y values
    """
    accurate = 0
    num_y = len(y)
    for i in range(0,num_y):
        if predicted_y[i] == y[i]:
            accurate += 1
    true_prop = float(accurate) / float(num_y)
    return true_prop


#Run the model
predictions = []
for i in range(NTREES):
    # We select 60% of the rows from train, sampling with replacement
    # We set a random state to ensure we'll be able to replicate our results
    # We set it to i instead of a fixed value so we don't get the same sample every time
    bag = cars_train.sample(frac=BAG_PROP, replace=True, random_state=i)
    
    # Fit a decision tree model to the "bag"
    clf = generate_tree(cars_train)
    clf.fit(bag[COLNAMES], bag["high_income"])
    
    # Using the model, make predictions on the test data
    predictions.append(clf.predict_proba(cars_test[COLNAMES])[:,1])

combined = numpy.sum(predictions, axis=0) / 10
rounded = numpy.round(combined)
