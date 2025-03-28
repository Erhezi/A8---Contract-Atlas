from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from models import User, USERS

auth_blueprint = Blueprint('auth', __name__, template_folder='templates')

@auth_blueprint.route('/landing')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    return render_template('landing.html')

@auth_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
        
    if request.method == 'POST':
        user_id = request.form['username']
        password = request.form['password']
        user = User.get(user_id)
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('main.index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@auth_blueprint.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
        
    if request.method == 'POST':
        user_id = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if user_id in USERS:
            flash('Username already exists', 'danger')
        elif password != confirm_password:
            flash('Passwords do not match', 'danger')
        else:
            # Add the new user to our simple storage
            USERS[user_id] = {'id': user_id, 'password': password}
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('auth.login'))
            
    return render_template('register.html')

@auth_blueprint.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.landing'))
