from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/sakthi')
def sakthi():
    return 'Hello, sakthi!'



if __name__ == '__main__':
    app.run(debug=True)


