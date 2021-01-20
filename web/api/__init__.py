from flask import Blueprint

apiBp = Blueprint('api',__name__)

from web.api import routes