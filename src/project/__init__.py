import os
from flask import Flask
from flask_graphql import GraphQLView

from project.views.health import health_blueprint
from project.schema import schema


def register_blueprints(app):
    app.register_blueprint(health_blueprint)


def create_app(script_info=None):
    app = Flask(__name__)

    app_settings = os.getenv('APP_SETTINGS')
    app.config.from_object(app_settings)
    app.add_url_rule(
        '/graphql',
        view_func=GraphQLView.as_view(
            'graphql',
            schema=schema,
            graphiql=True  # for having the GraphiQL interface
        )
    )

    register_blueprints(app)

    @app.shell_context_processor
    def ctx():
        return {'app': app}

    return app
