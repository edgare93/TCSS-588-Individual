from keras.models import load_model
import pandas as ps
import numpy as np

#Specify which set of models analysis will be performed on
attempt = 1

#Import the dataset which corresponds with that model
gene_file = "aml.data.RNA.reduced.csv"
#gene_file = "aml.data.RNA.5k.csv"
#gene_file = "aml.data.RNA.2.5k.csv"
#gene_file = "aml.data.RNA.1k.csv"
#gene_file = "aml.data.RNA.100.csv"
gene_data = ps.read_csv(gene_file, delimiter=',', index_col=0)

#Initialize some empty lists which will become the arrays of summed weights
unnormalized_out = []
normalized_out = []

#Iterate over the folds of the chosen model
for i in range(0,5):
    #Load the file using the appropriate values to specify the filename
    model = load_model("fold" + str(i) + "attempt" + str(attempt) + ".h5")

    #Get the weights of the input layer
    weights = model.layers[0].get_weights()[0]

    #Sum the absolute values of the weights for each input
    weight_sum = [sum(abs(x)) for x in weights]
    #Add those values to the unnormalized output array
    unnormalized_out.append(weight_sum)

    #Find the max and min values for normalization purposes
    max_value = max(weight_sum)
    min_value = min(weight_sum)

    #Apply the normalization and add those to the normalized output array
    weight_sum = [(x - min_value) / (max_value - min_value) for x in weight_sum]
    normalized_out.append(weight_sum)

#Convert both output arrays into pandas dataframes using the column labels from the input data
unnormalized_out = ps.DataFrame(unnormalized_out, columns=gene_data.columns.values.tolist())
normalized_out = ps.DataFrame(normalized_out, columns=gene_data.columns.values.tolist())

#Sanity check
print(unnormalized_out)
print(normalized_out)

#Save them as csv files, filenames are set manually
unnormalized_out.to_csv("unnormalized_values_10k.csv")
normalized_out.to_csv("normalized_values_10k.csv")
