from flask_restplus import Namespace, fields


class ClassificationDto:
    api = Namespace('user', description='user related operations')
    user = api.model('user', {
        'text': fields.String(required=True, description='user email address'),
        'sentiment': fields.String(required=True, description='user username'),
        'password': fields.String(required=True, description='user password'),
        'public_id': fields.String(description='user Identifier')
    })