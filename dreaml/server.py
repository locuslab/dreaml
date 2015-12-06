from threading import Thread
from frontend.start import create_app
from dreaml import DataFrame
import webbrowser

def start(df):
    """Launches a flask webapp to serve data visualizations for the
    dataframe df. By default this launches at http://localhost:5000"""
    if hasattr(df,"web_thread"):
        print "Server already running!"
    else:
        f = lambda a: create_app(a).run(debug=True,use_reloader=False)
        df.web_thread = Thread(target=f, args=(df,))
        df.web_thread.start()
        webbrowser.open("http://localhost:5000")

def _start_empty():
    df = DataFrame()
    f = lambda a: create_app(a).run()
    df.web_thread = Thread(target=f, args=(df,))
    df.web_thread.start()

    