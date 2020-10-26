# Alexandre Farias Portfolio

![image](https://images.unsplash.com/photo-1527474305487-b87b222841cc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1267&q=80)
*Image Source: [Unsplash](https://unsplash.com/photos/1K6IQsQbizI)*

Welcome to my Portfolio Page, my name is Alexandre Farias and I'm a Data Scientist with a Master Degree in Computational Intelligence from Universidade Federal do Pará (UFPA). I have a big interest in Natural Language Processing, Business Intelligence and Medical Research.\
My experience includes a good knowledge on Traditional Machine Learning and Deep Learning, working with Python, R, Scala and MATLAB, where I've developed CRM systems, Natural Language Processing (NLP) applications, Sports Analysis, Recommendation Systems, Forecasting Models and Medical Predictive Systems.\
Beyond Machine Learning, I have experience with containers (Docker), Nature Inspired Optimization Algorithms, Databases (SQL and NoSQL), Web Scraping and DevOps Culture.\
In this Portfolio you can see my Personal Projects and Accolades.

## Data Science Projects

* **Customer Churn Prediction**\
An implementation of customer churn prediction using Boosting Ensemble of Logistic Regressions. Also, an Exploratory Data Analysis is made to understand the impact of contracts types and payment methods on churned clients.\
This work was added to the Open Source Python Module PyCaret as one of the [examples on its official repository](https://github.com/pycaret/pycaret/blob/master/examples/PyCaret%202%20Customer%20Churn%20Prediction.ipynb).\
[Notebook on Nbviewer](https://nbviewer.jupyter.org/github/pycaret/pycaret/blob/master/examples/PyCaret%202%20Customer%20Churn%20Prediction.ipynb) - [Github Repository](https://github.com/alfarias/customer-churn-prediction)

* **Public Bids Anomaly Detection**\
This work has as objetive make an analysis on the public bids of the Brazilian State Rio Grande do Sul.
An Anomaly Detection Model is built to identify suspicious items bought on bids.\
The datasets contain text and numerical features about the bids description and items bought for the years of 2016 to 2019.\
For the analysis, the text features are analyzed for specific recurrent set of words, like items and public organ names, and for numerical features, the item costs.\
The anomaly detection is made using the iforest algorithm and later a classification with XGBoost.\
[Notebook on Nbviewer](https://nbviewer.jupyter.org/github/alfarias/rs-public-bids/blob/master/notebooks/bids_anomaly_detection.ipynb) - [Github Repository](https://github.com/alfarias/rs-public-bids)


* **Text Similarity Classification**\
Text similarity classification where an Exploratory Data Analysis is made to analyze character length effects on target classes and Topic Modelling using LDA to extract features from texts, to use they on a Bagging Ensemble of CatBoost Classifiers to check if the texts are similars. \
This work was added to the Open Source Python Module PyCaret as one of the [examples on its official repository](https://github.com/pycaret/pycaret/blob/master/examples/PyCaret%202%20Text%20Similarity%20Classification.ipynb).\
[Notebook on Nbviewer](https://nbviewer.jupyter.org/github/alfarias/text-similarity-classification/blob/master/notebooks/main.ipynb) - [Github repository](https://github.com/alfarias/text-similarity-classification)

* **HuffPost News Classification**\
A category classification of the news posted in HuffPost using their Headlines and Short Descriptions. The classification is made using [DistillBERT](https://arxiv.org/abs/1910.01108) pre-trained Deep Learning network.\
To help on the data understanding for model build, an Exploratory data analysis is made to analyze characters by news cateogry and the news posted through the years on HuffPost. \
This work was developed using the Python Deep Learning Frameworks [PyTorch](https://pytorch.org/) and [Catalyst](https://github.com/catalyst-team/catalyst).\
[Kaggle Notebook](https://www.kaggle.com/alfarias/huffpost-news-classification-with-distilbert) - [Github Repository](https://github.com/alfarias/news-classification-distilbert)

* **Fewer Injuries, More Touchdowns: NFL Data Analytics**\
This work was my contribution to [NFL 1st and Future - Analytics Competition](https://www.kaggle.com/c/nfl-playing-surface-analytics) hosted on Kaggle, where many features are analyzed, as Turf, Speed, etc., to determine how a player gets an injury.\
A highlight on this work is my take on reconstruct the `PlayKey` missing values, where I analyzed how the feature was was built based on other features. \
All this is only based on Data Analysis with Data features distribution by categories, features histograms, player's routes and features correlation.\
[Kaggle Notebook](https://www.kaggle.com/alfarias/fewer-injuries-more-touchdowns-data-analytics) - [Github Repository](https://github.com/alfarias/nfl-injuries-analytics)

* **Movie Recommendation System with ALS in PySpark**\
It's a Movie Recommendation System using [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) from Kaggle.\ The objective of this work is show how PySpark is effective to build recommendation systems with Collaborative Filtering. \
An Exploratory Data Analysis is made to get insights about the dataset and the system is built using Alternating Least Squares (ALS) algorithm.\
The recommendations are showed as User Based and Item based.\
[Kaggle Notebook](https://www.kaggle.com/alfarias/movie-recommendation-system-with-als-in-pyspark) - [Github Repository](https://github.com/alfarias/pyspark-movie-recommendation-system)

* **Brazilian Products Export Forecasting**\
Forecasting for exported brazilian products using the Prophet Python module developed by Facebook.\
This work uses covariates as countries GDPs to aid on the forecasting accuracy for the next 10 years.\
An Exploratory Data Analysis is made to gain insights about the exported products, as what country is a good trader.\
[Notebook on Nbviewer](https://nbviewer.jupyter.org/github/alfarias/forecasting-challenge-4i/blob/master/notebooks/case2.ipynb) - [Github Repository](https://github.com/alfarias/forecasting-challenge-4i)

* **Country TFP Forecasting**\
Forecasting for Country TFP using the Prophet Python module developed by Facebook. No covariates are used on this work. \
[Notebook on Nbviewer](https://nbviewer.jupyter.org/github/alfarias/forecasting-challenge-4i/blob/master/notebooks/case1.ipynb) - [Github Repository](https://github.com/alfarias/forecasting-challenge-4i)

* **Titanic Survivor Prediction with AutoML Tools**\
A project to show how fastai can give a bunch of tools to work in Python, how Pandas Profiling can speed up the EDA and the use of H2OAutoML to build many models.\
The Dataset used is the classical [Titanic Disaster](https://www.kaggle.com/c/titanic).\
[Kaggle Notebook](https://www.kaggle.com/alfarias/fastanic-fastai-pandas-profiling-h2o-automl) - [Github Repository](https://github.com/alfarias/titanic_survivor_h2oautoml)

* **MNIST with PyTorch/Catalyst + AMP(NVIDIA Apex)**\
This experiment is for the [MNIST Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) from Kaggle, the main objective is show how the use PyTorch and NVIDIA Apex for Mixed Precision Training.\
[Kaggle Notebook](https://www.kaggle.com/alfarias/mnist-with-pytorch-catalyst-amp-nvidia-apex) - [Github Repository](https://github.com/alfarias/digit-recognizer-catalyst-nvidia-apex)

* **MLP Optimization with Genetic Algorithm**\
This was a In Class Project on my Master Degree and the objective is use Genetic Algorithm to define the Hyperparameters (as neurons on the hidden layer, activation functions, etc.) of a Multi Layer Perceptron (MLP).\
Content in Brazilian Portuguese.\
[GitLab Project](https://gitlab.com/alfarias/ann-arrhythmia)

* **A Competitive Structure of Convolutional Autoencoder Networks for Electrocardiogram Signals Classification**\
Paper published by me and Adriana Castro on ENIAC 2018, where Convolutional Autoencoders are used in parallalel to classify Arrhythmia in ECGs. The Projec was developed with the Python Module Keras.
Content in Brazilian Portuguese.\
[Paper](https://sol.sbc.org.br/index.php/eniac/article/view/4446) - [GitLab Project](https://gitlab.com/alfarias/cae)

## Data Science Resources Compilation

* **Awesome Kaggle Kernels**\
This is a curated compilation of Kaggle Kernels to aid Data Scientists in their learning journey.\
I review and update it weekly. Contains Python and R works.\
[Github Repository](https://github.com/alfarias/awesome-kaggle-kernels)

## Data Analysis Dashboards

* **Telco Customer Churn**\
Customer Churn Analysis made on Tableau based on this [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn).\
It's an extension from Customer Churn Prediction Project.\
[Tableau Dashboard](https://public.tableau.com/profile/alexandre.farias#!/vizhome/Telco-Customers/ChurnDashboard)

## Nature Inspired Optmization Algorithms Projects

* **Genetic Algorithm Framework** \
A framework for Genetic Algorithm (GA) written in Python. Is the base for posterior works with GA made by me.\
Content in Brazilian Portuguese.\
[Presentation](https://raw.githubusercontent.com/alfarias/framework-ga/master/Apresenta%C3%A7%C3%A3o%20-%20Arcabou%C3%A7o%20do%20AG.pdf) - [Github Repository](https://github.com/alfarias/framework-ga)

* **Prisoner's dilemma**
Genetic Algorithm for the search of the a good solution for the Prisoner's Dilemma.\
Made from my Genetic Algorithm Framework with adaptations for this project.
Content in Brazilian Portuguese.\
[Presentation](https://gitlab.com/alfarias/ga_dilemadosprisioneiros/-/blob/master/apresentacao_-_dilema_dos_prisioneiros.pdf) - [GitLab Project](https://gitlab.com/alfarias/ga_dilemadosprisioneiros)

## Published Papers

* BAIA, A. F.; CASTRO, A. R. G . A Competitive Structure of Convolutional Autoencoder Networks for Electrocardiogram Signals Classification. In: Encontro Nacional de Inteligência Artificial e Computacional (ENIAC), 2018, São Paulo. Proceedings [...]. 2018. p.538-549.

* BAIA, A. F.; OLIVEIRA, S. R. B.; PEREIRA, G. T. M.; ALCANTARA, A. S.; QUARESMA, J. A. S.; RODRIGUES, E. A.; COSTA, I. E. F.; MORAES, H. R. S.. EXSCRUM - A Software Development Process Based on Practices Included in Agile Management and Engineering Methods. 2018. In: 15th CONTECSI - International Conference on Information Systems and Technology Management,
Proceedings [...]. 2018.
