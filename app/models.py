from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
from .extensions import db

class Carte(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    titlu = db.Column(db.String(255), nullable=False)
    autor = db.Column(db.String(255), nullable=False)
    an = db.Column(db.Integer, nullable=False)
    comentarii = db.relationship('Comentariu', backref='carte', lazy=True)

    def __repr__(self):
        return f"<Carte {self.titlu} de {self.autor}, anul {self.an}>"

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()  

def import_books_from_file(file_path):
    books = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith("Titlu:"):
                    parts = line.split(',')
                    title = parts[0].split(': ')[1].strip()
                    author = parts[1].split(': ')[1].strip()
                    year = int(parts[2].split(': ')[1].strip())
                    books.append(Carte(titlu=title, autor=author, an=year))
        db.session.bulk_save_objects(books)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Failed to import books: {str(e)}")

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    faculty = db.Column(db.String(100))
    year = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    profile_picture = db.Column(db.String(200))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def __repr__(self):
        return f'<User {self.username}>'

class Comentariu(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    continut = db.Column(db.Text, nullable=False)
    id_carte = db.Column(db.Integer, db.ForeignKey('carte.id'), nullable=False)
    id_utilizator = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    def __repr__(self):
        return f'<Comentariu "{self.continut[:30]}..." pentru Carte ID {self.id_carte}>'
