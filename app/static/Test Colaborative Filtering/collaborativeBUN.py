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