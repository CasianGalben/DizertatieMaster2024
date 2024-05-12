from app import create_app
from app.models import import_books_from_file

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        import_books_from_file('data/Date.txt')
    app.run()