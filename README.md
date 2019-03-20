# TCSS-588-Individual

Edgar Elliott's Individual Code submission.

# Files

## T-test selection.ipynb 
is the file that takes in the table of gene expression data produced by Nicole's preprocessing file (Which is too large to upload to github but was derived from the file found here: https://github.com/NCBI-Hackathons/ConsensusML/blob/master/Clinical_Data/AML_assay_clinical.csv), and outputs a set of data with fewer genes. It does this by using the labels to divide the patients into the two classes and performing a t-test for each gene between those two classes. It then reduces the table to genes which had a t-test result above some threshold. I wrote the original version which  used a threshold of 2 to cut the number of genes roughly in half. Later Alex modified it to use different thresholds to produce different dataset sizes.

## NeuralNetwork.py
is the file I used to create the Neural Network models. By changing the "attempt" number and input file it will train 5 NN's using 5-fold cross validation and save them to .h5 files with unique names. Since NN's are an unstable learner I have included the files for the networks which were generated for this project in a folder in this repository.

## ModelAnalysis.py
is the file that takes the saved NN models and extracts their weights, outputting both normalized and unnormalized versions into csv files. It works the same way as the NeuralNetwork file, requring the "attempt" value and input file to match the ones used when creating the Neural Networks. The output files are named manually.

# Dependencies

T-test selection has no package dependancies, however it does rely on Nicole's label and data files.

The two Python files require Tensorflow, Keras, Sklearn, Numpy, and Pandas. NeuralNetwork.py requires the output files from the T-test notebook and the label file. ModelSelection.py requires the same files used by NeuralNetwork.py as well as the .h5 files it generates and knowledge of the values that were used when running that file.
