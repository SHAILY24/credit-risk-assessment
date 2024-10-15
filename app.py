from flask import Flask, render_template
import pickle
import pandas as pd
from dotenv import dotenv_values
import os
env = dotenv_values()

app = Flask(__name__)

@app.route('/')
def home():
    # Load the saved results
    if os.path.exists('saved_results.pkl'):
        with open('saved_results.pkl', 'rb') as f:
            results = pickle.load(f)
        metrics = pd.DataFrame(results['metrics'])
    else:
        return "Please run generate_results.py first"

    # Data for the template
    data = {
        'metrics': metrics,
        'author': env['AUTHOR'],
        'email': env['EMAIL'],
        'linkedin': env['LINKEDIN']
    }

    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(port=int(env['PORT']))