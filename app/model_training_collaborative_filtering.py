import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV
import pickle
import os 
import numpy as np


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow, directory='app/static/images'):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = int((nCol + nGraphPerRow - 1) / nGraphPerRow)  
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plot_filename = os.path.join(directory, 'per_column_distribution.png')
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename


def plotCorrelationMatrix(df, graphWidth, directory='app/static/images'):
    filename = 'correlation_matrix.png'
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corr = df.corr()
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix', fontsize=15)
    plt.savefig(os.path.join(directory, filename))
    plt.close()
    return os.path.join(directory, filename)

def plotScatterMatrix(df, plotSize, textSize, directory='app/static/images'):
    filename = 'scatter_matrix.png'
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis='columns')
    df = df[[col for col in df if df[col].nunique() > 1]]
    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.savefig(os.path.join(directory, filename))
    plt.close()
    return os.path.join(directory, filename)

def plot_graphics(df, directory='app/static/images'):
    plot_filenames = {
        'distribution': plotPerColumnDistribution(df, 10, 5, directory),
        'correlation': plotCorrelationMatrix(df, 8, directory),
        'scatter': plotScatterMatrix(df, 9, 10, directory)
    }
    return plot_filenames


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip("'")
    if not {'userID', 'songID', 'rating'}.issubset(df.columns):
        raise ValueError("CSV-ul nu contine coloanele necesare.")
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['userID', 'songID', 'rating']], reader)

def hyperparameter_tuning(data):
    param_grid = {
        'n_epochs': [5, 10, 20],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.1, 0.5]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    return gs.best_params['rmse'], gs.best_score['rmse']

def train_and_save_model(file_path, save_path='static/collaborative_filtering.pkl'):
    data = load_dataset(file_path)
    best_params, best_scores = hyperparameter_tuning(data)

    algo = SVD(n_epochs=best_params['n_epochs'],
               lr_all=best_params['lr_all'],
               reg_all=best_params['reg_all'])

    trainset = data.build_full_trainset()
    algo.fit(trainset)

    evaluation_results = cross_validate(algo, data, measures=['rmse', 'mae'], cv=5, verbose=True)

    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f_out:
        pickle.dump(algo, f_out)

    return evaluation_results, save_path

def load_and_plot(file_path):
    df = pd.read_csv(file_path, delimiter=',', nrows=1000)
    df.dataframeName = os.path.basename(file_path)
    plot_graphics(df)
    return df.dataframeName 