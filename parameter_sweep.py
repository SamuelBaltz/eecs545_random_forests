#Import libraries
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier

bag_props = [x / 10.0 for x in range(1, 10, 1)]
train_props = [x / 10.0 for x in range(1, 10, 1)]
bias_switch = [True, False]

for bag_prop in bag_props:
    for train_prop in train_props:
        for bias_on_off in bias_switch:
            #Initialize constants
            TRAIN_PROP = train_prop
            SEED = 545
            NPARRAY = 0     #1 if the data should be an array; 0 if pandas df
            NSTATE = 1
            DEPTH = 5
            NLEAF = 2
            NTREES = 100
            BAG_PROP = bag_prop
            NCORRECTS = 3
            BIAS_CHECKING = bias_on_off
            DATASET = 'income'
            
            if BIAS_CHECKING and DATASET == 'cars':
                print("\n \n WARNING: cannot bias check on categorical variable \n \n")
                BREAK
            
            #set the random seed
            np.random.seed(SEED)
            
            
            #Import data
            if DATASET == 'cars':
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
                cars = cars.replace('vgood',3)
                cars = cars.replace('good',2)
                cars = cars.replace('acc',1)
                cars = cars.replace('unacc',0)
            
            
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
            
            if DATASET == 'income':
                # read in data
                income = pd.read_csv("income.csv")
            
                COLNAMES = list(income.columns.values)    #Save column names to a list   
            
                # convert categorical variable to numerical variable
                for col in COLNAMES:
                    income[col] = pd.Categorical(income[col]).codes
                COLNAMES.remove('y')
            
                #  shuffle, split into training and testing set
                income = income.reindex(np.random.permutation(income.index)).iloc[:1700,:]
                train_max_row = int(math.floor(income.shape[0] * TRAIN_PROP))
                income_train = income.iloc[:train_max_row]
                income_test = income.iloc[train_max_row:]
            
                #Lazy assignment to not have to rewrite everything else
                cars_train = income_train
                cars_test = income_test
            
            def generate_tree(bag):
                global NSTATE
                global NLEAF
                global COLNAMES
                tree = DecisionTreeClassifier(random_state=NSTATE,min_samples_leaf=NLEAF,  \
                                                 splitter = 'random', max_features = 'auto')
                tree.fit(bag[COLNAMES], bag["y"])
                return tree
            
            
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
            
            #Run the model
            def grow_forest(train_data,test_data,NTREES,bias_checking,is_b,estimated_bias):
                global COLNAMES
                global DATASET
            
                if DATASET == 'cars':
                    if bias_checking:
                        votes = np.zeros((len(train_data),4))
                    if not bias_checking:
                        votes = np.zeros((len(test_data),4))
            
                if DATASET == 'income':
                    if bias_checking:
                        votes = np.zeros((len(train_data),2))
                    if not bias_checking:
                        votes = np.zeros((len(test_data),2))
            
                for i in range(NTREES):
                # We select BAG_PROP of the rows from train, sampling with replacement
                # We set a random state to ensure we'll be able to replicate our results
                # We set it to i instead of a fixed value so we don't get the same sample every time
                    if not bias_checking:
                        bag = sample_with_replacement(train_data,i,bias_checking)
                    if bias_checking:
                        (bag,sample_list) = sample_with_replacement(train_data,i,bias_checking)
                        unsampled_list = np.setdiff1d(train_data.index,sample_list)
                    
                    if not is_b:
                        # Fit a decision tree model to the "bag"
                        tree = generate_tree(bag)
                        tree.fit(bag[COLNAMES], bag["y"])
                    if is_b:
                        tree = generate_tree(train_data)
                        tree.fit(train_data[COLNAMES],estimated_bias)
            
                    if bias_checking:
                        # Using the model, make predictions on the training data
                        probabilities = tree.predict_proba(train_data[COLNAMES])
                        prob1 = probabilities[:,0]
                        prob2 = probabilities[:,1]
                        if DATASET == 'cars':
                            prob3 = probabilities[:,2]
                            prob4 = probabilities[:,3]
                            for j in range(0,len(votes)):
                                row_probs = (prob1[j],prob2[j],prob3[j],prob4[j])
                                votes[j,np.argmax(row_probs)] += 1
                        if DATASET == 'income':
                            for j in range(0,len(votes)):
                                row_probs = (prob1[j],prob2[j])
                                votes[j,np.argmax(row_probs)] += 1
            
                    if not bias_checking:
                        # Using the model, make predictions on the test data
                        probabilities = tree.predict_proba(test_data[COLNAMES])
                        prob1 = probabilities[:,0]
                        prob2 = probabilities[:,1]
                        if DATASET == 'cars':
                            prob3 = probabilities[:,2]
                            prob4 = probabilities[:,3]
                            for j in range(0,len(votes)):
                                row_probs = (prob1[j],prob2[j],prob3[j],prob4[j])
                                votes[j,np.argmax(row_probs)] += 1
                        if DATASET == 'income':
                            for j in range(0,len(votes)):
                                row_probs = (prob1[j],prob2[j])
                                votes[j,np.argmax(row_probs)] += 1
            
                predicted_y = []
                for i in range(0,len(votes)):
                    predicted_y.append(np.argmax(votes[i]))
            
                if bias_checking:
                    return (predicted_y,sample_list,unsampled_list)
                if not bias_checking:
                    return predicted_y
            
            def check_prediction(y, predicted_y):
                """
                Check the accuracy of a predicted y column against the real y values
                """
                global BIAS_CHECKING
                accurate = np.zeros((len(y),1))
                estimated_bias = np.zeros((len(y),1))
                y_array = np.array(y)
                for i in range(len(accurate)):
                    if predicted_y[i] == y_array[i]:
                        accurate[i] = 1
                    estimated_bias[i] = predicted_y[i] - y_array[i]
                if BIAS_CHECKING:
                    return (accurate,estimated_bias)
                else:
                    return accurate
            
            def correct_bias(predicted_y,sample_list,unsampled_list,train_data):
                (accurate,estimated_bias) = check_prediction(train_data['y'],predicted_y)
                #Grow another RF with training data and response variable of estimated_bias
                (b_predicted_y,b_sample_list,b_unsampled_list) = grow_forest(cars_train,cars_test,NTREES,BIAS_CHECKING,True,estimated_bias)
                #Recover the accuracy and bias of the new random forest
                (b_accurate,b_estimated_bias) = check_prediction(train_data['y'],b_predicted_y)
                corrected_y = predicted_y[0] - b_estimated_bias + 1
                (b_accurate,null) = check_prediction(train_data['y'],corrected_y)
                #oob_error = sum(accurate)[0]/float(len(accurate))
                return (b_accurate)
            
            if BIAS_CHECKING:
                (predicted_y,sample_list,unsampled_list) = grow_forest(cars_train,cars_test,NTREES,BIAS_CHECKING,False,-999)
                accurate = correct_bias(predicted_y,unsampled_list,sample_list,cars_train)
                accuracy_rate = str(sum(accurate)[0]/float(len(accurate)))
                print("accurate: " + str(accuracy_rate))
                
            
            if not BIAS_CHECKING:
                predicted_y = grow_forest(cars_train,cars_test,NTREES,BIAS_CHECKING,False,-999)
                accuracy_rate = str(float(sum(check_prediction(cars_test['y'],predicted_y))) / float(len(cars_test)))
                print("accurate: " + accuracy_rate)
            
            f = open("data_to_plot.csv", "a+")
            f.write('\n' + str(accuracy_rate) + ',' + str(bag_prop) + ',' + str(train_prop) + ',' + str(bias_on_off) + ',' + str(DATASET))
            f.close()





#combined = np.sum(predicted_y, axis=0) / float(NTREES)
#rounded = np.round(combined)
#mse = sum((rounded - cars_test['y'])**2)/float(len(cars_test['y']))
