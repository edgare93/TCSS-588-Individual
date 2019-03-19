import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import KFold
import pandas as ps


#Change this number so the models don't save over one another
attempt = 1

#Import one of the five reduced datasets
gene_file = "aml.data.RNA.reduced.csv"
#gene_file = "aml.data.RNA.5k.csv"
#gene_file = "aml.data.RNA.2.5k.csv"
#gene_file = "aml.data.RNA.1k.csv"
#gene_file = "aml.data.RNA.100.csv"
gene_data = ps.read_csv(gene_file, delimiter=',', index_col=0)

#Import the class label file
label_file = "aml.data.labels.csv"
label_data = ps.read_csv(label_file, delimiter=',', index_col=0)

#Set up the 5-fold cross validation
kf = KFold(n_splits=5)

#Iterate over the train and test indeces for the different folds
i = 0
for train_index, test_index in kf.split(gene_data, y=label_data):

    #isolate the training and test data from the gene file
    X_train = gene_data.ix[train_index]
    X_test = gene_data.ix[test_index]

    #Isolate the training and test data from the label file
    y_train = label_data.ix[train_index]
    y_test = label_data.ix[test_index]

    #Convert them to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    #Construct the model
    model = Sequential()
    model.add(Dense(64))
    model.add(Activation('softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #I set the learning rate extremely low because early on the model couldn't learn anything other than the baseline
    opt = Adam(lr = 0.00001)

    #Compile the model
    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              #optimizer='adam',
              metrics=['accuracy'])

    #Fit the model to the training and validation data, using early stopping to prevent overfitting
    model.fit(
        X_train,
        y_train,
        callbacks=[EarlyStopping(patience=5)],
        epochs=100,
        validation_data=(X_test, y_test)
        )

    #Save the model to a file using the fold number and the value set at the start to create the filename
    name = 'fold' + str(i) + 'attempt' + str(attempt) + '.h5'
    i += 1
    model.save(name)
