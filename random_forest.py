#Import libraries
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


#Initialize constants
TRAIN_PROP = .2
SEED = 545
NPARRAY = 0     #1 if the data should be an array; 0 if pandas df
NSTATE = 1
DEPTH = 5
NLEAF = 2
NTREES = 10
BAG_PROP = .6
NCORRECTS = 3
BIAS_CHECKING = True

#set the random seed
np.random.seed(SEED)


#Import data
cars = pd.read_csv('car.csv')

COLNAMES = list(cars.columns.values)    #Save column names to a list
COLNAMES.remove('y')

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


#If NPARRAY = 1, rewrite the pandas dataframe as a np array so that we can
    # treat it the same way we've been treating other datasets throughout this
    # semester
if NPARRAY == 1:
    cars = cars.values

#Split into training and test data
    #The original dataset is sorted by the value of the first column, so we
    # need to shuffle the order of the rows of the array or else our training
    # data will only contain specific values of one variable
cars = cars.reindex(np.random.permutation(cars.index))
train_max_row = int(math.floor(cars.shape[0] * TRAIN_PROP))
cars_train = cars.iloc[:train_max_row]
cars_test = cars.iloc[train_max_row:]
#np.random.shuffle(cars)
#cars_train = cars[0:NTRAIN]
#cars_test = cars[NTRAIN:len(cars)]

def generate_tree(bag):
    global NSTATE
    global NLEAF
    global COLNAMES
    tree = DecisionTreeClassifier(random_state=NSTATE,min_samples_leaf=NLEAF,  \
                                     splitter = 'random', max_features = 'auto')
    tree.fit(bag[COLNAMES], bag["y"])
    return tree


def predict(data):
    global COLNAMES
    prediction = tree.predict(data[COLNAMES])



def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or np array.
    """
    # Compute the counts of each unique value in the column
    counts = np.bincount(column)
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
    median = np.median(column)
    
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

def sample_with_replacement(data,i,bias_checking):
    """
    Create a sample with replacement for the bootstrap aggregation process
    """
    global BAG_PROP
    bag = pd.DataFrame.sample(data,n=int(len(data)*BAG_PROP),   \
                                                   replace=True, random_state=i)
    sample_indices = list(bag.index)
        
    if not bias_checking:
        return bag

    if bias_checking:
        return(bag,sample_indices)

def check_prediction(y, predicted_y):
    """
    Check the accuracy of a predicted y column against the real y values
    """
    accurate = 0
    num_y = len(y)
    y_array = np.array(y)
    for i in range(0,num_y):
        if predicted_y[i] == y_array[i]:
            accurate += 1
    true_prop = float(accurate) / float(num_y)
    return true_prop

#Run the model
def grow_forest(train_data,test_data,NTREES,bias_checking):
    global COLNAMES

    if bias_checking:
        votes = np.zeros((len(train_data),4))
    if not bias_checking:
        votes = np.zeros((len(test_data),4))

    for i in range(NTREES):
    # We select BAG_PROP of the rows from train, sampling with replacement
    # We set a random state to ensure we'll be able to replicate our results
    # We set it to i instead of a fixed value so we don't get the same sample every time
        if not bias_checking:
            bag = sample_with_replacement(train_data,i,bias_checking)
        if bias_checking:
            (bag,sample_list) = sample_with_replacement(train_data,i,bias_checking)
            unsampled_list = np.setdiff1d(train_data.index,sample_list)
        
        # Fit a decision tree model to the "bag"
        tree = generate_tree(bag)
        tree.fit(bag[COLNAMES], bag["y"])

        if bias_checking:
            # Using the model, make predictions on the training data
            probabilities = tree.predict_proba(train_data[COLNAMES])
            prob1 = probabilities[:,0]
            prob2 = probabilities[:,1]
            prob3 = probabilities[:,2]
            prob4 = probabilities[:,3]
            for j in range(0,len(votes)):
                row_probs = (prob1[j],prob2[j],prob3[j],prob4[j])
                votes[j,np.argmax(row_probs)] += 1

        if not bias_checking:
            # Using the model, make predictions on the test data
            probabilities = tree.predict_proba(test_data[COLNAMES])
            prob1 = probabilities[:,0]
            prob2 = probabilities[:,1]
            prob3 = probabilities[:,2]
            prob4 = probabilities[:,3]
            for j in range(0,len(votes)):
                row_probs = (prob1[j],prob2[j],prob3[j],prob4[j])
                votes[j,np.argmax(row_probs)] += 1

    predicted_y = []
    for i in range(0,len(votes)):
        predicted_y.append(np.argmax(votes[i]) + 1)

    if bias_checking:
        return (predicted_y,sample_list,unsampled_list)
    if not bias_checking:
        return predicted_y

def correct_bias(predicted_y,sample_list,unsampled_list,train_data):
    accurate = check_prediction(train_data['y'],predicted_y)
    return accurate

if BIAS_CHECKING:
    (predicted_y,sample_list,unsampled_list) = grow_forest(cars_train,cars_test,NTREES,BIAS_CHECKING)
    accurate = correct_bias(predicted_y,unsampled_list,sample_list,cars_train)
    print accurate

if not BIAS_CHECKING:
    predicted_y = grow_forest(cars_train,cars_test,NTREES,BIAS_CHECKING)
    print(check_prediction(cars_test['y'],predicted_y))




#combined = np.sum(predicted_y, axis=0) / float(NTREES)
#rounded = np.round(combined)
#mse = sum((rounded - cars_test['y'])**2)/float(len(cars_test['y']))
