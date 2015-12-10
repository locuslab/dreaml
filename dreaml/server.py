from threading import Thread
from frontend.start import create_app
from dreaml import DataFrame
import webbrowser

def start(df,new=1):
    """Launches a flask webapp to serve data visualizations for the
    dataframe df. By default this launches at http://localhost:5000"""
    if hasattr(df,"web_thread"):
        print "Server already running!"
    else:
        f = lambda a: create_app(a).run(debug=True,use_reloader=False)
        df.web_thread = Thread(target=f, args=(df,))
        df.web_thread.start()
        if new>=0:
            webbrowser.open("http://localhost:5000",new=new)    

def _start_empty():
    df = DataFrame()
    f = lambda a: create_app(a).run()
    df.web_thread = Thread(target=f, args=(df,))
    df.web_thread.start()

    