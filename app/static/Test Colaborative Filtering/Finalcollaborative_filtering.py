
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt #
import numpy as np 
import os 
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, GridSearchCV
import pickle

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] 
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = int((nCol + nGraphPerRow - 1) / nGraphPerRow)  
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()



def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna(axis='columns') 
    df = df[[col for col in df if df[col].nunique() > 1]] 
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()




def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) 
    df = df.dropna(axis = 'columns')
    df = df[[col for col in df if df[col].nunique() > 1]] 
    columnNames = list(df)
    if len(columnNames) > 10: 
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


nRowsRead = 1000 

df1 = pd.read_csv('songsDataset.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'songsDataset.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 9, 10)


import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, GridSearchCV

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
    print("Available measures in best_params:", gs.best_params.keys())  
    return gs.best_params, gs.best_score

def evaluate_algorithm(algo, data):
    results = cross_validate(algo, data, measures=['rmse', 'mae'], cv=5, verbose=True)
    return results

file_path = 'songsDataset.csv'
data = load_dataset(file_path)
best_params, best_scores = hyperparameter_tuning(data)

algo = SVD(n_epochs=best_params['rmse']['n_epochs'],
           lr_all=best_params['rmse']['lr_all'],
           reg_all=best_params['rmse']['reg_all'])

trainset = data.build_full_trainset()
algo.fit(trainset)


evaluation_results = evaluate_algorithm(algo, data)
print("Rezultatele evaluării:", evaluation_results)


model_file_path = 'collaborative_filtering.pkl'
with open(model_file_path, 'wb') as f_out:
    pickle.dump(algo, f_out)
print(f"Modelul a fost salvat ca {model_file_path}.")

