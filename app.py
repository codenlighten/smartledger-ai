import os
from flask import Flask, request, jsonify, render_template
from agent import main_agent

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    user_request = request.json['user_request']
    api_key = request.json['api_key']
    satisfied = request.json['satisfied']
    executed_tasks = main_agent(user_request, api_key, satisfied)
    return jsonify({"executed_tasks": executed_tasks})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))





