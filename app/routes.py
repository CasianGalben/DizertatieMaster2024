from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for, session, current_app, send_from_directory
from werkzeug.utils import secure_filename
from .models import db, Carte, User, Comentariu
from sqlalchemy.orm import joinedload
from .model_training_collaborative_filtering import train_and_save_model, load_and_plot
from .model_training_Named_Entity_Rec_training import run_training
import os
import numpy as np


main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/cartiAfisare', methods=['GET'])
def get_carti():
    carti = Carte.query.all()
    return jsonify([{'titlu': carte.titlu, 'autor': carte.autor, 'an': carte.an} for carte in carti])

@main.route('/carti')
def show_carti():
    carti = Carte.query.all()
    return render_template('carti.html', carti=carti)

@main.route('/login')
def login():
    return render_template('login.html')

@main.route('/proceseaza_login', methods=['POST'])
def proceseaza_login():
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
    if user:
        session['user_id'] = user.id
        session['username'] = user.username
        flash('Ai fost conectat cu succes!', 'success')
        return redirect(url_for('main.profile'))
    else:
        flash('Utilizator inexistent.', 'danger')
        return redirect(url_for('main.login'))

@main.route('/profile')
def profile():
    user= None
    return render_template('profile.html', user=user)

@main.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])

    if request.method == 'POST':
        if user is None:
            user = User()  
            db.session.add(user)

        user.first_name = request.form['first_name']
        user.last_name = request.form['last_name']
        user.faculty = request.form['faculty']
        user.year = request.form.get('year', type=int)
        user.gender = request.form['gender']

        file = request.files['profile_picture']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('path_to_your_upload_folder', filename)
            file.save(filepath)
            user.profile_picture = filepath

        db.session.commit()
        flash('Profilul a fost actualizat cu succes.', 'success')

    return render_template('profile_update.html', user=user if user else {})

@main.route('/BunVenit')
def bun_venit():
    if 'user_id' in session:
        return redirect(url_for('main.profile'))
    else:
        return redirect(url_for('main.login'))

@main.route('/search')
def search():
    query = request.args.get('query', '')
    search_pattern = f"%{query}%"
    matching_books = Carte.query.options(joinedload(Carte.comentarii)).filter(
        db.or_(
            Carte.titlu.like(search_pattern),
            Carte.autor.like(search_pattern),
            db.cast(Carte.an, db.String).like(search_pattern) 
        )
    ).all()
    return render_template('search_results.html', books=matching_books, query=query)

@main.route('/carte/<int:id_carte>', methods=['GET', 'POST'])
def detalii_carte(id_carte):
    carte = Carte.query.get_or_404(id_carte)
    comentarii = Comentariu.query.filter_by(id_carte=id_carte).all()
    if request.method == 'POST':
        action = request.form.get('action', 'Adauga')  

        if action == 'Adauga':
            continut_comentariu = request.form['continut']
            comentariu_nou = Comentariu(
                continut=continut_comentariu,
                id_carte=id_carte,
                id_utilizator=None  
            )
            db.session.add(comentariu_nou)
            db.session.commit()
        elif action.startswith('Sterge:'):
            comentariu_id = int(action.split(':')[1])  
            comentariu = Comentariu.query.get_or_404(comentariu_id)
            db.session.delete(comentariu)
            db.session.commit()

        return redirect(url_for('main.detalii_carte', id_carte=id_carte))
    
    return render_template('detalii_carte.html', carte=carte, comentarii=comentarii)

@main.route('/train-collaborative-filtering', methods=['POST'])
def train_collaborative_filtering():
    file = request.files['dataset']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        evaluation_results, model_path = train_and_save_model(file_path)
        graphics_filenames = load_and_plot(file_path)
        session['model_path'] = model_path
        session['graphics_filenames'] = graphics_filenames
        flash('Model antrenat și grafice generate cu succes.')
        return redirect(url_for('main.display_results_collaborative'))
    else:
        flash('Eroare la încărcarea fișierului.')
        return redirect(url_for('main.home'))
    
from flask import render_template, session, jsonify
import json

@main.route('/display-results-collaborative', methods=['GET'])
def display_results_collaborative():
    evaluation_results = session.get('evaluation_results', {})
    model_path = session.get('model_path', '')
    graphics_filenames_json = session.get('graphics_filenames', '{}')

    
    try:
        graphics_filenames = json.loads(graphics_filenames_json)
    except json.JSONDecodeError:
        graphics_filenames = {}

    return render_template('results_collaborative.html',
                           evaluation_results=evaluation_results,
                           model_path=model_path,
                           graphics_filenames=graphics_filenames)


@main.route('/get-graph/<filename>', methods=['GET'])
def get_graph(filename):
    directory = os.path.join(current_app.config['UPLOAD_FOLDER'], 'images')
    return send_from_directory(directory, filename)

@main.route('/upload_collaborative_data', methods=['POST'])
def upload_collaborative_data():
    file = request.files['data_file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        
        evaluation_results, model_path = train_and_save_model(filepath)
        graphics_filenames = load_and_plot(filepath)
        
        
        for key, value in evaluation_results.items():
            if isinstance(value, np.ndarray):
                evaluation_results[key] = value.tolist()
        
        
        session['evaluation_results'] = evaluation_results
        session['model_path'] = model_path
        session['graphics_filenames'] = graphics_filenames
        
        flash('Modelul a fost antrenat și graficele au fost generate cu succes.')
        return redirect(url_for('main.display_results_collaborative'))
    else:
        flash('Nu a fost selectat niciun fișier', 'error')
        return redirect(url_for('main.index'))
    
@main.route('/download_collaborative_model/<filename>')
def download_collaborative_model(filename):
    directory = 'E:/Dizertatie/static'
    if not os.path.exists(os.path.join(directory, filename)):
        return f"Fișierul {filename} nu a fost găsit.", 404
    
    return send_from_directory(directory=directory, path=filename, as_attachment=True)


@main.route('/upload_NER_file', methods=['POST'])
def upload_NER_file():
    if 'data_file' not in request.files:
        flash('No file part')
        return redirect(url_for('main.home'))
    
    file = request.files['data_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('main.home'))
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        
        results = run_training(file_path)
        session['NER_results'] = results
        flash('NER model trained successfully.')
        return redirect(url_for('main.display_NER_results'))

    return redirect(url_for('main.home'))

@main.route('/display_NER_results')
def display_NER_results():
    results = session.get('NER_results', {})
    return render_template('display_NER_results.html', results=results)

@main.route('/download_NER_model')
def download_NER_model():
    directory = 'E:/Dizertatie/static' 
    filename = 'NER_MODEL.pth'  
    if not os.path.exists(os.path.join(directory, filename)):
        flash('Modelul nu a fost găsit.', 'error')
        return redirect(url_for('main.display_NER_results'))
    
    return send_from_directory(directory=directory, path=filename, as_attachment=True)


