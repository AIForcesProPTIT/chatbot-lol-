from flask import Blueprint

# apiBp = Blueprint('api',__name__)

messBp = Blueprint('test_socket',__name__)

from web.socket_test import routes

# from web.api import routes