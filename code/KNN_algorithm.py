import numpy as np

class KnnModel:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train #Train data
        self.y_train = y_train #train labels
        self.m, self.n = x_train.shape #Store the dimentions of training features.
        
    def predict(self, x_test):
        self.x_test = x_test #test set to predict the values
        self.m_test, self.n = x_test.shape # Store the dimentions of test set.
        y_predict = np.zeros(self.m_test) #Create an array to save the predicted values
        
        for i in range(self.m_test): # loop through each row of the test set.
            x = self.x_test[i]
            
            neighbors = np.zeros(self.k) #create an array to save the neighbors around the data point.
            neighbors = self.find_neighbors(x) # Now it is time to specify the exact points around the test point.
            y_predict[i] = np.min(neighbors) #the Minimum distance amoung tthe neighbors is our predicted value.
            
        return y_predict
            
    def find_neighbors(self, x):
        eucl_distance = np.zeros(self.m) #create the an array to store the euclidean calculation.
        
        for i in range(self.m):
            d = self.calculate_distance(x, self.x_train[i]) # calculate the distance.
            eucl_distance[i] = d
            
        index = eucl_distance.argsort() #Return the indices that would sort an array.
        y_train_sorted = self.y_train[index] 

        return y_train_sorted[:self.k] # According to how many neighbors we had specified return sorted y_train.
            
    def calculate_distance(self, x, x_train):
        distance = np.sqrt(np.sum(np.square(x - x_train))) # implement the exact equation.
        
        return distance
