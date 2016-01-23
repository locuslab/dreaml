from flask import Blueprint, render_template, flash, redirect, url_for, Markup
from flask_bootstrap import __version__ as FLASK_BOOTSTRAP_VERSION
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
from markupsafe import escape

from flask_nav import Nav
from flask_nav.elements import Navbar, View

# from bokeh.embed import autoload_server, components

nav = Nav()

def construct_frontend(df=None):
    frontend = Blueprint('frontend', __name__)


    @nav.navigation()
    def mynavbar():
        return Navbar(
            Link('CILO','/'),
            View('DataFrame Grid', 'frontend.grid'),
            View('Computational Graph', 'frontend.graph'),
            View('Plots', 'frontend.plots'),
        )


    # Our index-page just shows a quick explanation. Check out the template
    # "templates/index.html" documentation for more details.
    @frontend.route('/')
    def grid():
        return render_template('grid.html')

    @frontend.route('/graph')
    def graph():
        return render_template('graph.html')

    @frontend.route('/plots')
    def plots():
        scripts = []
        # divs_markedup = []
        if df is not None and len(df._plots)>0:  
            # script, divs = components(df._plots)
            # scripts.append(Markup(script))
            scripts = df._plots
            #     raise ValueError
            scripts = [Markup(s) for s in scripts]
        return render_template('plots.html',
                               scripts=scripts)
                               # divs=divs_markedup)

    @frontend.route('/json/structure')
    def json_structure():
        if df is not None:
            return df.structure_to_json()
        else:
            return '{}'

    @frontend.route('/json/graph')
    def json_graph():
        if df is not None:
            return df.graph_to_cytoscape_json()
        else:
            return '{}'

    return frontend