import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, GridSearchCV

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip("'")
    
    if not {'userID', 'songID', 'rating'}.issubset(df.columns):
        raise ValueError("CSV does not contain required columns.")
        
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['userID', 'songID', 'rating']], reader)

def evaluate_algorithm(algo, data):
    return cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=2, verbose=True)  

def hyperparameter_tuning(data):
    param_grid = {
        'n_epochs': [5, 10],  
        'lr_all': [0.005, 0.01],  
        'reg_all': [0.02, 0.1]  
    }
    gs = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=2)  
    gs.fit(data)
    # Correctly access the results
    best_params = gs.best_params['RMSE']
    best_score = gs.best_scores['RMSE']
    print("Available measures in results:", gs.best_scores.keys())
    return best_params, best_score



file_path = 'songsDataset.csv'

data = load_dataset(file_path)


print("Evaluating SVD with default parameters...")
default_algo = SVD()
evaluate_algorithm(default_algo, data)


print("Performing hyperparameter tuning...")
best_params, best_score = hyperparameter_tuning(data)
print("Best Parameters:", best_params)
print("Best RMSE Score:", best_score['RMSE'])


algo = SVD(n_epochs=best_params['RMSE']['n_epochs'], lr_all=best_params['RMSE']['lr_all'],
           reg_all=best_params['RMSE']['reg_all'])
trainset = data.build_full_trainset()
algo.fit(trainset)
print("Model trained with best parameters.")
