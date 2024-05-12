from flask import Blueprint, request, jsonify, redirect, url_for, session, render_template, send_from_directory
import pandas as pd
from io import StringIO
import os
from .model_training import load_and_preprocess_data, train_naive_bayes, evaluate_model, save_model


naive_bayes_blueprint = Blueprint('naive_bayes', __name__)

@naive_bayes_blueprint.route('/upload_NaiveBayes_data', methods=['POST'])
def upload_NaiveBayes_data():
    train_data = request.files['train_data']
    test_data = request.files['test_data']
    train_df = pd.read_csv(StringIO(train_data.read().decode('utf-8')))
    test_df = pd.read_csv(StringIO(test_data.read().decode('utf-8')))
    trained_grid, X_test, y_test = train_naive_bayes(train_df)
    results = evaluate_model(trained_grid, X_test, y_test)
    
    model_filename = save_model(trained_grid.best_estimator_)
    results['Model File'] = model_filename  # Stocare fișier în rezultate pentru folosire
    
    # Salvăm rezultatele
    session['results'] = results
    return redirect(url_for('naive_bayes.display_results'))

@naive_bayes_blueprint.route('/results', methods=['GET'])
def display_results():
    results = session.get('results', {})
    return render_template('results.html', results=results)

@naive_bayes_blueprint.route('/download_model/<filename>')
def download_model(filename):
    directory = os.path.abspath(os.path.dirname(__file__))  
    #Luăm ruta proiectului pentru a putea lua modelul generat
    model_directory = os.path.join(directory, '..')  
    return send_from_directory(directory=model_directory, path=filename, as_attachment=True)
