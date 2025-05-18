# Authentication blueprint and user management
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from ..models import User

# Create the blueprint
auth_blueprint = Blueprint('auth', __name__, 
                          url_prefix='/auth',
                          template_folder='templates')

@auth_blueprint.route('/landing')
def landing():
    """Main landing page"""
    if current_user.is_authenticated:
        return redirect(url_for('common.dashboard'))
    return render_template('landing.html')

@auth_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for('common.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple validation
        if not username or not password:
            flash('Please enter both username and password', 'danger')
            return render_template('login.html')
        
        # Try to authenticate
        success, message = User.check_password(username, password)
        if success:
            user = User.get_by_username(username)
            login_user(user)

            user_id = user.id

            current_step_key = f'current_step_id_{user_id}'
            completed_steps_key = f'completed_steps_{user_id}'
            
            # Initialize workflow state
            if current_step_key not in session:
                session[current_step_key] = 1
            if completed_steps_key not in session:
                session[completed_steps_key] = []
                session.modified = True
            
            # Redirect to intended page or dashboard
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('common.dashboard'))
        else:
            flash(message, 'danger')
            
    return render_template('login.html')

@auth_blueprint.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect(url_for('common.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        name = request.form.get('name', '')
        
        # Simple validation
        if not username or not password or not email:
            flash('All fields are required', 'danger')
            return render_template('register.html')
        
        # Check if user already exists
        if User.get_by_username(username):
            flash('Username already exists', 'danger')
            return render_template('register.html')
        
        # Create user
        password_hash = generate_password_hash(password)
        user = User(username=username, email=email, name=name, password_hash=password_hash)
        
        if user.save():
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash('Error creating user account. Please try again.', 'danger')
            
    return render_template('register.html')

@auth_blueprint.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    # Clear session data
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.landing'))