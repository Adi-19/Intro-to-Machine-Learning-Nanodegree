# Intro to Machine Learning Nanodegree
# Projects
# Project 1: Supervised Learning - @Finding Donors for CharityML

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly 15 million working Californians, CharityML has brought you on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail. Your goal will be evaluate and optimize several different supervised learners to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent.

. Preprocessed data by transforming skewed functions, normalizing/scaling, and one-hot encoding
. Evaluated models based on metrics such as Accuracy, Precision, Recall, F Beta Score, as well as compare them against a Naive Predictor
. Explored real-world application of models, advantages, and disadvantages
. Evaluated Supervised Learning techniques including Gaussian Naive Bayes, Decision Trees, Support Vector Machines (SVM) and chooses the best model
. Created training and predicting pipeline to quickly and efficiently train models using various sizes of training sets and perform predictions on testing data
. Tuned model using Grid Search to optimize hyperparameters
. Extracted feature importance
. Performed feature selection to get the most important features and re-trained the model
. Examined effects of feature selection by comparing the model with reduced features and optimized model

# Project 2: Deep Learning - Image Classifier

Developed an image classifier with PyTorch, then converted it into a command line application.

. Loaded training data, validation data, testing data, label mappings, and applied transformations (random scaling, cropping, resizing, flipping) to training data
. Normalized means and standard deviations of all image color channels, shuffled data and specified batch sizes
. Loaded pre-trained VGG16 network
. Defined a new untrained feed-forward network as a classifier, using ReLU activations, and Dropout
. Defined Negative Log-Likelihood Loss, Adam Optimizer, and learning rate
. Trained the classifier layers with backpropagation in a CUDA GPU using the pre-trained network to ~90% accuracy on the validation set
. Graphed training/validation/testing loss and validation/testing accuracy to ensure convergence to a global (or sufficient local) minimum
. Saved and loaded model to perform inference later
. Preprocessed images (resize, crop, normalized means and standard deviations) to use as input for model testing
. Visually displayed images to ensure preprocessing was successful
. Predicted the class/label of an image using the trained model and plotted top 5 classes to ensure the validity of the prediction

# Project 3: Unsupervised Learning - Identify Customer Segments

Applied unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that I used has been provided by Udacity's partners at Bertelsmann Arvato Analytics and represents a real-life data science task.

. Assessed missing data and converted missing value codes to NaNs
. Identified, and dropped features/samples that were outliers (features missing more than 20% of data)
. Performed data wrangling and re-encoded categorical (via One-hot Encoding) and mixed features
. Used an Imputer to replace all missing values
. Applied feature scaling (via StandardScaler)
. Applied PCA to find vectors of maximal variance and reduce dimensionality
. Analyzed the ratio of explained variance accounted for by each principal component and decided to retain 20 principal components for clustering
. Re-fitted a PCA instance on the determined transformation and reviewed the cumulative variance
. Interpreted principal components to determine the most prominent features
. Applied clustering (via KMeans) to the general population and used the Elbow Method to decide how many clusters to keep, then re-fit the K-means model with the selected number   of clusters
. Mapped the customer data to the clusters for the general population (pre-processed, transformed features, applied clustering via PCA and KMeans from the general population,     and obtained cluster predictions for the customer demographic)
. Compared customer demographic to the general population to see where the strongest customer base for the company is
. Discovered ~200 features of people that are a suitable target audience for the company (also discovered 4 groups/clusters of people that aren't a suitable target audience for   the company)
