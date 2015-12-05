from threading import Thread
from frontend.start import create_app
from dreaml import DataFrame

def start(df):
    if hasattr(df,"web_thread"):
        print "Server already running!"
    else:
        f = lambda a: create_app(a).run(debug=True,use_reloader=False)
        df.web_thread = Thread(target=f, args=(df,))
        df.web_thread.start()

def start_empty():
    df = DataFrame()
    f = lambda a: create_app(a).run()
    df.web_thread = Thread(target=f, args=(df,))
    df.web_thread.start()

    