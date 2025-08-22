from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user

# Create the blueprint
admin_blueprint = Blueprint('admin', __name__, 
                          url_prefix='/admin',
                          template_folder='templates')