import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir
from pathlib import Path
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from db import Build_Database, getPredictions, processResults, getClassifications, processResultsC, plot_confusion_matrix, getPredictionsConfusionMatrix
from pyts.image import GramianAngularField
import cv2
import matplotlib.pyplot as plt
from shutil import copyfile
import time
import glob
from PIL import Image
import base64
import io
import datetime
from flask import Flask, request, render_template_string
from flask import Flask, render_template_string
from flask_sqlalchemy import SQLAlchemy
from flask_user import current_user, login_required, roles_required, UserManager, UserMixin


# Class-based application configuration
class ConfigClass(object):
    """ Flask application config """

    # Flask settings
    SECRET_KEY = 'This is an INSECURE secret!! DO NOT use this in production!!'

    # Flask-SQLAlchemy settings
    SQLALCHEMY_DATABASE_URI = 'sqlite:///basic_app.sqlite'    # File-based SQL database
    SQLALCHEMY_TRACK_MODIFICATIONS = False    # Avoids SQLAlchemy warning

    # Flask-Mail SMTP server settings
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 465
    MAIL_USE_SSL = True
    MAIL_USE_TLS = False
    MAIL_USERNAME = 'rajeevratan722@gmail.com'
    MAIL_PASSWORD = 'Cookie123.'
    MAIL_DEFAULT_SENDER = '"MyApp" <rajeevratan722@gmail.com'

    # Flask-User settings
    USER_APP_NAME = "Material Identification Tool"      # Shown in and email templates and page footers
    USER_ENABLE_EMAIL = True        # Enable email authentication
    USER_ENABLE_USERNAME = False    # Disable username authentication
    USER_EMAIL_SENDER_NAME = USER_APP_NAME
    USER_EMAIL_SENDER_EMAIL = "rajeevratan722@gmail.com"

def create_app():
    
    app = Flask(__name__)
    app.config.from_object(__name__+'.ConfigClass')
    
    # Build initial database
    BD = Build_Database(k = 15) # k = 15
    #df, labels, orig_data_labels_train, key_hash_train = BD.initializeDB("uploads")
    #BD.createIndex(df)

    # Initialize Flask-SQLAlchemy
    db = SQLAlchemy(app)

    # Define the User data-model.
    # NB: Make sure to add flask_user UserMixin !!!
    class User(db.Model, UserMixin):
        __tablename__ = 'users'
        id = db.Column(db.Integer, primary_key=True)
        active = db.Column('is_active', db.Boolean(), nullable=False, server_default='1')

        # User authentication information. The collation='NOCASE' is required
        # to search case insensitively when USER_IFIND_MODE is 'nocase_collation'.
        email = db.Column(db.String(255, collation='NOCASE'), nullable=False, unique=True)
        email_confirmed_at = db.Column(db.DateTime())
        password = db.Column(db.String(255), nullable=False, server_default='')

        # User information
        first_name = db.Column(db.String(100, collation='NOCASE'), nullable=False, server_default='')
        last_name = db.Column(db.String(100, collation='NOCASE'), nullable=False, server_default='')

        # Define the relationship to Role via UserRoles
        roles = db.relationship('Role', secondary='user_roles')

    # Define the Role data-model
    class Role(db.Model):
        __tablename__ = 'roles'
        id = db.Column(db.Integer(), primary_key=True)
        name = db.Column(db.String(50), unique=True)

    # Define the UserRoles association table
    class UserRoles(db.Model):
        __tablename__ = 'user_roles'
        id = db.Column(db.Integer(), primary_key=True)
        user_id = db.Column(db.Integer(), db.ForeignKey('users.id', ondelete='CASCADE'))
        role_id = db.Column(db.Integer(), db.ForeignKey('roles.id', ondelete='CASCADE'))

    # Setup Flask-User and specify the User data-model
    user_manager = UserManager(app, db, User)

    # Create all database tables
    db.create_all()

    # Create 'member@example.com' user with no roles
    if not User.query.filter(User.email == 'member@example.com').first():
        user = User(
            email='member@example.com',
            email_confirmed_at=datetime.datetime.utcnow(),
            password=user_manager.hash_password('Password1'),
        )
        db.session.add(user)
        db.session.commit()

    # Create 'admin@example.com' user with 'Admin' and 'Agent' roles
    if not User.query.filter(User.email == 'admin@example.com').first():
        user = User(
            email='admin@example.com',
            email_confirmed_at=datetime.datetime.utcnow(),
            password=user_manager.hash_password('Password1'),
        )
        user.roles.append(Role(name='Admin'))
        user.roles.append(Role(name='Agent'))
        db.session.add(user)
        db.session.commit()

    # The Home page is accessible to anyone
    @app.route('/')
    def home_page():
        # String-based templates
        return render_template_string("""
            {% extends "flask_user_layout.html" %}
            <link rel="stylesheet" href="/static/style.css">
            {% block content %}
                <center>
                <img class="mb-4" src="/static/logo.png" alt="" width="272">
                <p><a href={{ url_for('predictor') }}>Input Data to Classifier</a></p>
                <p><a href={{ url_for('confusion_mat') }}>Generate Confusion Matrix</a></p>
                <p><a href={{ url_for('get_results') }}>Use Siamese Network</a></p>
                <p><a href={{ url_for('upload_file') }}>Add to Database</a></p>
                <p><a href={{ url_for('user.register') }}>Register</a></p>
                <p><a href={{ url_for('user.login') }}>Sign in</a></p>
                <p><a href={{ url_for('home_page') }}>Home page</a> </p>
                <p><a href={{ url_for('user.logout') }}>Sign out</a></p>
                </center>
            {% endblock %}
            """)
    #<p><a href={{ url_for('get_results') }}>Input Data </a></p>
    @app.route('/add', methods=['GET', 'POST'])
    @roles_required('Admin')
    def upload_file():
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files.get('file')
            if not file:
                return
            #img_bytes = file.read()
            file.save(secure_filename(file.filename))
            
            copyfile(file.filename, f'uploaded/{file.filename}')
            time.sleep(0.1)
            #folder = os.path.join(app.instance_path, 'uploads') # Assigns upload path to variable
            #print(folder)
            #file.save('uploaded/upload.xlsx')

            #class_id, class_name = get_prediction(image_bytes=img_bytes)
            #class_name = format_class_name(class_name)
            BD.addData(f'uploaded/{file.filename}')
            df, labels, orig_data_labels_train, key_hash_train = BD.initializeDB("uploads")
            BD.convertToImagesAdd(df, orig_data_labels_train, labels)
            BD.movetoAve()
            
            # get filename of recently uploaded 
            
            
            #return render_template('result.html', class_id=class_id,
            #                       class_name=class_name)
        return render_template('add.html')

    # Use Siamese Network
    @app.route('/samples', methods=['GET', 'POST'])
    def get_results():
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files.get('file')
            if not file:
                return
            #img_bytes = file.read()
            new_filename = secure_filename(file.filename)
            file.save(new_filename)
            
            files = glob.glob('tests/*')
            for f in files:
                os.remove(f)
            
            copyfile(new_filename, f'tests_uploads/{new_filename}')
            
            BD.addData(f'tests_uploads/{new_filename}', addorread = 'read')
            df_test, test_labels, orig_data_labels_test, key_hash_test = BD.loadQuery("tests")
            BD.convertToImages(df_test, orig_data_labels_test, test_labels)
            results = getPredictions()
            #results.to_csv('results.csv')
            print('[INFO] Results here')
            print(results)
            final_results = processResults(results)
            final_results.reset_index(inplace=True)
            print('FINAL RESULTS')
            print(final_results)
            #final_results.to_csv('final_results.csv')
            #D, I = BD.search(df_test)
            #predictions = BD.getLabel(D, I, orig_data_labels_test)        
            #predictions.to_csv('predictions.csv')
            #print(predictions)
            
            return render_template("result.html", column_names=final_results.columns.values,
                                   row_data=list(final_results.values.tolist()),
                                    link_column="Sample", zip=zip)
            #return render_template('result.html', class_id=predictions)
        return render_template('samples.html')

    # Use Classifier
    @app.route('/classifier', methods=['GET', 'POST'])
    def predictor():
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files.get('file')
            if not file:
                return
            #img_bytes = file.read()
            new_filename = secure_filename(file.filename)
            file.save(new_filename)
            
            files = glob.glob('tests/*')
            for f in files:
                os.remove(f)
            
            copyfile(new_filename, f'tests_uploads/{new_filename}')
            
            BD.addData(f'tests_uploads/{new_filename}', addorread = 'read')
            df_test, test_labels, orig_data_labels_test, key_hash_test = BD.loadQuery("tests")
            BD.convertToImages(df_test, orig_data_labels_test, test_labels)
            
            # Use Siamese Network
            results = getClassifications()
            print(results)
            final_results = processResultsC(results)
            
            #D, I = BD.search(df_test)
            #predictions = BD.getLabel(D, I, orig_data_labels_test)        
            #predictions.to_csv('predictions.csv')
            #print(predictions)
            
            return render_template("result.html", column_names=final_results.columns.values,
                                   row_data=list(final_results.values.tolist()),
                                    link_column="Sample", zip=zip)
            #return render_template('result.html', class_id=predictions)
        return render_template('classifier.html')
   
   
    @app.route('/confusion_matrix', methods=['GET', 'POST'])
    def confusion_mat():
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            file = request.files.get('file')
            if not file:
                return
            #img_bytes = file.read()
            new_filename = secure_filename(file.filename)
            file.save(new_filename)
            
            files = glob.glob('tests/*')
            for f in files:
                os.remove(f)
            
            copyfile(new_filename, f'tests_uploads/{new_filename}')
            
            BD.addData(f'tests_uploads/{new_filename}', addorread = 'read')
            df_test, test_labels, orig_data_labels_test, key_hash_test = BD.loadQuery("tests")
            BD.convertToImages(df_test, orig_data_labels_test, test_labels)
            
            results = getPredictionsConfusionMatrix()
            print(test_labels)
            print(results)
            conf_mat = confusion_matrix(test_labels, results)
            target_names = list(range(0,24))
            plot_url = plot_confusion_matrix(conf_mat, target_names)
            #return render_template('confusion_matrix_results.html', name = 'new_plot', url ='new_plot.png')
            return render_template('confusion_matrix_results.html', plot_url=plot_url)
            
            # im = Image.open("cfm.png") #Open the generated image
            # data = io.BytesIO() 
            # im.save(data, "png")           
            # encoded_img_data = base64.b64encode(data.getvalue())

            # return render_template("confusion_matrix_results.html", img=encoded_img_data.decode('utf-8'))

            #return render_template('result.html', class_id=predictions)
        return render_template('confusion_matrix.html') 
    
    
    return app

# Start development web server
if __name__=='__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8080, debug=True)
    
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')

# @app.route('/add', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files.get('file')
#         if not file:
#             return
#         #img_bytes = file.read()
#         file.save(secure_filename(file.filename))
        
#         copyfile(file.filename, f'uploaded/{file.filename}')
#         time.sleep(0.1)
#         #folder = os.path.join(app.instance_path, 'uploads') # Assigns upload path to variable
#         #print(folder)
#         #file.save('uploaded/upload.xlsx')

#         #class_id, class_name = get_prediction(image_bytes=img_bytes)
#         #class_name = format_class_name(class_name)
#         BD.addData(f'uploaded/{file.filename}')
#         df, labels, orig_data_labels_train, key_hash_train = BD.initializeDB("uploads")
#         BD.createIndex(df)
        
#         # get filename of recently uploaded 
        
        
#         #return render_template('result.html', class_id=class_id,
#         #                       class_name=class_name)
#     return render_template('add.html')

# @app.route('/samples', methods=['GET', 'POST'])
# def get_results():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files.get('file')
#         if not file:
#             return
#         #img_bytes = file.read()
#         file.save(secure_filename(file.filename))
        
#         files = glob.glob('tests/*')
#         for f in files:
#             os.remove(f)
        
#         copyfile(file.filename, f'tests_uploads/{file.filename}')
        
#         BD.addData(f'tests_uploads/{file.filename}', addorread = 'read')
#         df_test, test_labels = BD.loadQuery("tests")
#         D, I = BD.search(df_test)
#         predictions = BD.getLabel(D, I)        
#         predictions.to_csv('predictions.csv')
#         print(predictions)
        
#         return render_template("result.html", column_names=predictions.columns.values,
#                                row_data=list(predictions.values.tolist()),
#                                 link_column="sample_number", zip=zip)
#         #return render_template('result.html', class_id=predictions)
#     return render_template('samples.html')

# if __name__ == '__main__':
#     app.run()