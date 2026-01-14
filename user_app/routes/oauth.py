from flask import Blueprint, redirect, url_for
from authlib.integrations.flask_client import OAuth

oauth_bp = Blueprint('oauth', __name__, url_prefix='/oauth')
oauth = OAuth()

google = oauth.register(
    name='google',
    client_id='...',
    client_secret='...',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    client_kwargs={'scope': 'openid profile email'}
)

@oauth_bp.route('/login/google')
def google_login():
    redirect_uri = url_for('oauth.google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@oauth_bp.route('/callback/google')
def google_callback():
    token = google.authorize_access_token()
    user_info = google.parse_id_token(token)
    return f"Hello {user_info['name']}"
