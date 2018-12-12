#Import libraries
import pandas as pd

#Import data
cars = pd.read_csv('car.csv')

#Reshape data into a numeric array
#Replace car text data cells as follows:
    #buying and maint variables
        #vhigh -> 3
        #high -> 2
        #med -> 1
        #low -> 0
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

#Rewrite the pandas dataframe as a numpy array so that we can treat it the same
    #way we've been treating other datasets throughout this semester
cars = cars.values

