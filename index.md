# Introduction

Welcome to my Portfolio Page, my name is Alexandre Farias and I'm a Data Scientist with a Master Degree in Computational Intelligence from Universidade Federal do Pará (UFPA). I have a big interest in Natural Language Processing, Business Intelligence and Medical Research. <br>
My experience includes a knowledge on Traditional Machine Learning and Deep Learning, with works using Python, R, Scala and MATLAB, where I've worked with CRM systems, Natural Language Processing (NLP), Sports Analysis, Recommendation Systems, Forecasting and Medical Predictive Systems.<br>
Beyond Machine Learning, I have a good knowledge about work with containers (Docker), Nature Inspired Optimization Algorithms, Databases (SQL and NoSQL), Web Scraping and DevOps Culture.<br>
In this page you can see my Personal Projects and Accolades.<br>

# Data Science Projects
* **Customer Churn Prediction**<br>
An implementation of customer churn prediction using Boosting Ensemble of Logistic Regressions. Also, an Exploratory Data Analysis is made to understand the impact of contracts types and payment methods on churned clients.<br>
This work was added to the Open Source Python Module PyCaret as one of the [examples on its official repository](https://github.com/pycaret/pycaret/blob/master/examples/PyCaret%202%20Customer%20Churn%20Prediction.ipynb).<br>
[Notebook on Nbviewer](https://nbviewer.jupyter.org/github/pycaret/pycaret/blob/master/examples/PyCaret%202%20Customer%20Churn%20Prediction.ipynb) - [Github Repository](https://github.com/alfarias/customer-churn-prediction)

* **Text Similarity Classification**<br>
Text similarity classification where an Exploratory Data Analysis is made to analyze character length effects on target classes and Topic Modelling using LDA to extract features from texts, to use they on a Bagging Ensemble of CatBoost Classifiers to check if the texts are similars. <br>
This work was added to the Open Source Python Module PyCaret as one of the [examples on its official repository](https://github.com/pycaret/pycaret/blob/master/examples/PyCaret%202%20Text%20Similarity%20Classification.ipynb).<br>
[Notebook on Nbviewer](https://nbviewer.jupyter.org/github/alfarias/text-similarity-classification/blob/master/notebooks/main.ipynb) - [Github repository](https://github.com/alfarias/text-similarity-classification)

* **HuffPost News Classification**<br>
A category classification of the news posted in HuffPost using their Headlines and Short Descriptions. The classification is made using [DistillBERT](https://arxiv.org/abs/1910.01108) pre-trained Deep Learning network.<br>
To help on the data understanding for model build, an Exploratory data analysis is made to analyze characters by news cateogry and the news posted through the years on HuffPost. <br>
This work is developed using the Python Deep Learning Frameworks [PyTorch](https://pytorch.org/) and [Catalyst](https://github.com/catalyst-team/catalyst).<br>
[Kaggle Notebook](https://www.kaggle.com/alfarias/huffpost-news-classification-with-distilbert) - [Github Repository](https://github.com/alfarias/news-classification-distilbert)


* **Fewer Injuries, More Touchdowns: NFL Data Analytics**<br>
This work was my contribution to [NFL 1st and Future - Analytics Competition]() hosted on Kaggle, where many features are analyzed, as Turf, Speed, etc., to determine how a player gets a injury.<br>
A highlight on this work is my take on reconstruct the `PlayKey` missing values, where I analyzed how the feature was was built based on other features. <br>
All this is only based on Data Analysis with Data features distribution by categories, features histograms, player's routes and features correlation.
[Kaggle Notebook](https://www.kaggle.com/alfarias/fewer-injuries-more-touchdowns-data-analytics) - [Github Repository](https://github.com/alfarias/nfl-injuries-analytics)

* **Movie Recommendation System with ALS in PySpark**<br>
It's a Movie Recommendation System using [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) from Kaggle.<br> The objective of this work is show how PySpark is effective to build recommendation systems with Collaborative Filtering. <br>
An Exploratory Data Analysis is made to get insights about the dataset and the system is built using Alternating Least Squares (ALS) algorithm.<br>
The recommendations are showed as User Based and Item based.<br>
[Kaggle Notebook](https://www.kaggle.com/alfarias/movie-recommendation-system-with-als-in-pyspark) - [Github Repository](https://github.com/alfarias/pyspark-movie-recommendation-system)

* **Brazilian Products Export Forecasting**<br>
Forecasting for exported brazilian products using the Prophet Python module developed by Facebook.<br>
[Notebook on Nbviewer](https://nbviewer.jupyter.org/github/alfarias/forecasting-challenge-4i/blob/master/notebooks/case2.ipynb) - [Github Repository](https://github.com/alfarias/forecasting-challenge-4i)

* **Country TFP Forecasting**<br>
Forecasting for Country TFP using the Prophet Python module developed by Facebook.<br>
[Notebook on Nbviewer](https://nbviewer.jupyter.org/github/alfarias/forecasting-challenge-4i/blob/master/notebooks/case1.ipynb) - [Github Repository](https://github.com/alfarias/forecasting-challenge-4i)

* **Titanic Survivor Prediction with AutoML Tools**<br>
A project to show how fastai can give a bunch of tools to work in Python, how Pandas Profiling can speed up the EDA and the use of H2OAutoML to build many models.<br>
The Dataset used is the classical [Titanic Disaster](https://www.kaggle.com/c/titanic).
[Kaggle Notebook](https://www.kaggle.com/alfarias/fastanic-fastai-pandas-profiling-h2o-automl) - [Github Repository](https://github.com/alfarias/titanic_survivor_h2oautoml)

* **MNIST with PyTorch/Catalyst + AMP(NVIDIA Apex)**<br>
This experiment is for the [MNIST Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) from Kaggle, the main objective is show how the use PyTorch and NVIDIA Apex for Mixed Precision Training.<br>
[Kaggle Notebook](https://www.kaggle.com/alfarias/mnist-with-pytorch-catalyst-amp-nvidia-apex) - [Github Repository](https://github.com/alfarias/digit-recognizer-catalyst-nvidia-apex)

* **MLP Optimization with Genetic Algorithm**<br>
This was a Project Study in Master Degree and the objective is use Genetic Algorithm to define the Hyperparameters of a Multi Layer Perceptron (MLP).<br>
Content in Brazilian Portuguese.<br>
[GitLab Project](https://gitlab.com/alfarias/ann-arrhythmia)

* **A Competitive Structure of Convolutional Autoencoder Networks for Electrocardiogram Signals Classification**<br>
Paper published by me and Adriana Castro on ENIAC 2018, where Convolutional Autoencoders are used in parallalel to classify Arrhythmia in ECGs. The Projec was developed with the Python Module Keras.
Content in Brazilian Portuguese.<br>
[Paper](https://sol.sbc.org.br/index.php/eniac/article/view/4446) - [GitLab Project](https://gitlab.com/alfarias/cae)

# Data Analysis Dashboards
* **Telco Customer Churn**<br>
Customer Churn Analysis made on Tableau based on this [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn).<br>
It's an extension from Customer Churn Prediction Project.
[Tableau Dashboard](https://public.tableau.com/profile/alexandre.farias#!/vizhome/Telco-Customers/ChurnDashboard)

# Nature Inspired Optmization Algorithms Projects
* **Genetic Algorithm Framework** <br>
A framework for Genetic Algorithm (GA) written in Python. Is the base for posterior works with GA made by me.<br>
Content in Brazilian Portuguese.<br>
[Presentation](https://raw.githubusercontent.com/alfarias/framework-ga/master/Apresenta%C3%A7%C3%A3o%20-%20Arcabou%C3%A7o%20do%20AG.pdf) - [Github Repository](https://github.com/alfarias/framework-ga)

* **Prisoner's dilemma**
Genetic Algorithm for the search of the a good solution from the Prisoner's dilemma.<br> Made from Genetic Algorithm Framework with adaptations for this project.
Content in Brazilian Portuguese.<br>
[Presentation](https://gitlab.com/alfarias/ga_dilemadosprisioneiros/-/blob/master/apresentacao_-_dilema_dos_prisioneiros.pdf) - [GitLab Project](https://gitlab.com/alfarias/ga_dilemadosprisioneiros)

# Published Papers

* BAIA, A. F.; CASTRO, A. R. G . A Competitive Structure of Convolutional Autoencoder Networks for Electrocardiogram Signals Classification. In: Encontro Nacional de Inteligência Artificial e Computacional (ENIAC), 2018, São Paulo. Proceedings [...]. 2018. p.538-549.

* BAIA, A. F.; OLIVEIRA, S. R. B.; PEREIRA, G. T. M.; ALCANTARA, A. S.; QUARESMA, J. A. S.; RODRIGUES, E. A.; COSTA, I. E. F.; MORAES, H. R. S.. EXSCRUM - A Software Development Process Based on Practices Included in Agile Management and Engineering Methods. 2018. In: 15th CONTECSI - International Conference on Information Systems and Technology Management,
Proceedings [...]. 2018.
