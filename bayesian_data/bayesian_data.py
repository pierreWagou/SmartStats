import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV

######################
# General functions #
#####################

# read_data
def read_data(filename):
    """ Read data from a CSV file and provide an overview through a plot
    Parameters
    ----------
    filename : str
        name of the file
    Returns
    -------
    DataFrame
        extracted data
    """
    data = pd.read_csv(filename)
    plot_data(data)
    return data

# plot_data
def plot_data(data):
    """ Plot data repartition via pair plotting
    Parameters
    ----------
    data : DataFrame
        data to be plotted
    """
    plot = sns.pairplot(data=data, hue='z')
    plt.savefig("pair_plot.png")
    plt.clf()

# create_partition
def create_partition(data):
    """ Create a training/test and features/class based partition
    Parameters
    ----------
    data : DataFrame
        data to be divided
    Returns
    -------
    tuple
        tuple containing training set (features and class) and test set (feature and class)
    """
    X = data.drop(['z'], axis=1)
    z = data['z']
    partition = train_test_split(X, z, train_size=0.66)
    return partition

# da
def compute_score(filename, model, param):
    """ Compute accuracy score from a classic given model
    Parameters
    ----------
    filename : str
        name of the file
    model : function
        training and testing of a specific model
    param: dict
        hyperparameter names and values
    Returns
    -------
    float
        accuracy score of the model
    """
    data = read_data(filename)
    partition = create_partition(data)
    score = model(partition, param)
    return score

# compare_da
def compare_model(filename, model1, model2, param1, param2):
    """ compare two classic given models
    Parameters
    ----------
    filename : str
        name of the file
    model1 : function
        training and testing of a specific model
    model2 : function
        training and testing of another model
    param1: dict
        hyperparameter names and values for first model
    param2: dict
        hyperparameter names and values for second model
    Returns
    -------
    float
        difference in accuracy score between the models
    """
    data = read_data(filename)
    scores = []
    for _ in range(100):
        partition = create_partition(data)
        score1 = model1(partition, param1)
        score2 = model2(partition, param2)
        scores.append(score1-score2)
    return np.mean(scores)

#######################################
# 1. Classical discriminant analysis #
######################################

# Question 1 #

def train_gda(X_train, z_train):
    """ train GDA
    Parameters
    ----------
    X_train : DataFrame
        sub set containing the training features
    z_train : DataFrame
        sub set containing the training label
    Returns
    -------
    dict
        model with the computed the empirical values
    """
    categories = list(set(z_train))
    prior = [np.mean(z_train==category) for category in categories]
    exp = [np.mean(X_train[z_train==category]) for category in categories]
    cov = [np.cov(X_train[z_train==category].T) for category in categories]
    model = {
        'categories': categories,
        'prior': prior,
        'exp': exp,
        'cov': cov
    }
    return model

def test(model, X_test):
    """ Test model
    Parameters
    ----------
    model : dict
        model with empirical values
    X_test : DataFrame
        sub set containing the testing features
    Returns
    -------
    list
        label predictions
    """
    nb_class = len(model['categories'])
    pdf = [multivariate_normal.pdf(X_test, model['exp'][i], model['cov'][i], allow_singular=True) for i in range(nb_class)]
    product = [model['prior'][i]*pdf[i] for i in range(nb_class)]
    post = np.array([np.divide(product[i], sum(product), out=np.zeros_like(product[i]), where=sum(product)!=0) for i in range(nb_class)]).T
    pred = [model['categories'][list(post[i]).index(max(post[i]))] for i in range(len(X_test))]
    return pred

# Question 2 #

def plot_predict(pred, X_test, z_test):
    """ Plot prediction to compare with reality
    Parameters
    ----------
    pred : list
        list of all the predicted label
    X_test : DataFrame
        sub set containing the testing features
    z_test : DataFrame
        sub set containing the testing classes
    """
    plot = sns.scatterplot(data=X_test, x='X1', y='X2', hue=pred, style=z_test)
    leg_handles = plot.get_legend_handles_labels()[0]
    plot.legend(leg_handles, ['pred1-test1', 'pred1-test2', 'pred2-test1', 'pred2-test2'], title='Legend')
    plot.set(xlabel="X1", ylabel="X2")
    plot.set_title("Prediction and reality")
    plt.savefig("pred.png")
    plt.clf()

def gda(partition, param=None):
    """ Train, test and get score from GDA
    Parameters
    ----------
    partition : tuple
        sub sets for trainin/testing and feature/class
    Returns
    -------
    float
        accuracy score of GDA
    """
    X_train, X_test, z_train, z_test = partition
    model = train_gda(X_train, z_train)
    pred = test(model, X_test)
    score = accuracy_score(z_test, pred)
    plot_predict(pred, X_test, z_test)
    return score

#compute_param
def compute_param(X, z, param):
    """ compute param with cross validation for C hyper parameter
    Parameters
    ----------
    solver : str
        name of solver
    penalty : str
        name of penalty
    param : dict
        incomplete params
    Returns
    -------
    dict
        hyperparameter names and values
    """
    param_grid = {'C': np.geomspace(10**-1, 10**1, 2)}
    poly = PolynomialFeatures(degree=2, include_bias=False)
    cv = GridSearchCV(LogisticRegression(max_iter=1000, solver=param['solver'], penalty=param['penalty']), param_grid, cv=3, scoring="accuracy")
    pipe = make_pipeline(poly, cv)
    pipe.fit(X, z)
    cv.fit(X, z)
    param['C'] = cv.best_params_['C']
    return param

# pqlr
def pqlr(partition, param):
    """ Train, test and get score from PQLR
    Parameters
    ----------
    partition : tuple
        sub sets for trainin/testing and feature/class
    param : dict
        hyperparameter names and values
    Returns
    -------
    float
        accuracy score of PQLR
    """
    X_train, X_test, z_train, z_test = partition
    if param['penalty']!='none':
        param = compute_param(X_train, z_train, param)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    model = LogisticRegression(max_iter=1000, C=param['C'], solver=param['solver'], penalty=param['penalty'])
    pipe = make_pipeline(poly, model)
    pipe.fit(X_train, z_train)
    pred = pipe.predict(X_test)
    score = accuracy_score(z_test, pred)
    return score

param_qlr = {
   'solver': 'lbfgs',
    'penalty': 'none',
    'C': 1
}

print('1. Classical discriminant analysis')
gda_score = compute_score('data/synth.csv', gda, None)
print(f'Accuracy score of GDA is {gda_score}')
qlr_score = compute_score('data/synth.csv', pqlr, param_qlr)
print(f'Accuracy score of QLR is {qlr_score}')
dif_score = compare_model('data/synth.csv', gda, pqlr, None, param_qlr)
print(f'Difference between GDA and QDA is {dif_score}')

#########################################
# 2. Regularized discriminant analysis #
########################################

# Question 3 #

# train_rda

def train_rda(X_train, z_train, param):
    """ train RDA
    Parameters
    ----------
    X_train : DataFrame
        sub set containing the training features
    z_train : DataFrame
        sub set containing the training label
    param : dict
        hyperparameter names and values
    Returns
    -------
    dict
        model with the computed the empirical values
    """
    mean = param['mean']
    shrink = param['shrink']
    scale = param['scale']
    df = param['df']
    categories = list(set(z_train))
    nb_class = len(categories)
    prior = [np.mean(z_train==category) for category in categories ]

    n = np.array([z_train[z_train==category].count() for category in categories])
    exp_naive = np.array([np.mean(X_train[z_train==category]) for category in categories])
    exp = [(n[i]*exp_naive[i]+shrink[i]*mean[i]) / (n[i]+shrink[i]) for i in range(nb_class)]

    cov_naive = np.array([np.cov(X_train[z_train==category].T) for category in categories])
    denom = n + df + nb_class + 2
    exp_dif = np.array([(exp_naive[i]-mean[i])*(exp_naive[i]-mean[i]).T for i in range(nb_class)])
    scale_inv = np.array([np.linalg.inv(scale[i]) for i in range(nb_class)])
    cov = [(n[i]*cov_naive[i] + (n[i]*shrink[i]/(n[i]+shrink[i])) * exp_dif[i] + scale_inv[i]) / denom[i] for i in range(nb_class)]
    model = {
        'categories': categories,
        'prior': prior,
        'exp': exp,
        'cov': cov
    }
    return model

# rda
def rda(partition, param):
    """ Train, test and get score from RDA
    Parameters
    ----------
    partition : tuple
        sub sets for trainin/testing and feature/class
    param : dict
        hyperparameter names and values
    Returns
    -------
    float
        accuracy score of RDA
    """
    X_train, X_test, z_train, z_test = partition
    model = train_rda(X_train, z_train, param)
    pred = test(model, X_test)
    score = accuracy_score(z_test, pred)
    return score

# Question 2 #

param_rda = {
    'mean': np.array([np.zeros(2), np.zeros(2)]),
    'shrink': [0, 0],
    'scale': [np.eye(2), np.eye(2)],
    'df': [0, 0]
}

param_pqlr = {
   'solver': 'lbfgs',
    'penalty': 'l2',
    'C':1
}

print('2. Regularized discriminant analysis')
rda_score = compute_score('data/synth.csv', rda, param_rda)
print(f'Accuracy score of RDA is {rda_score}')
pqlr_score = compute_score('data/synth.csv', pqlr, param_pqlr)
print(f'Accuracy score of PQLR is {pqlr_score}')
dif_score = compare_model('data/synth.csv', rda, pqlr, param_rda, param_pqlr)
print(f'Difference between RDA and PQLR is {dif_score}')
