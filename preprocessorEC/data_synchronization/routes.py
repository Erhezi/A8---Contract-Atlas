from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, Response
from flask import current_app, stream_with_context
from flask_login import login_required, current_user
from ..common.db import get_db_connection


data_synchronization_bp = Blueprint('data_synchronization', __name__,
                           url_prefix='/data-synchronization',
                           template_folder='templates')