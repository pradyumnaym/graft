from flask import Flask, send_from_directory

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

@app.route("/paper/<path:path>")
def paper_files(path):
    return send_from_directory("paper", path)

if __name__ == "__main__":
    app.run(port=8081, debug=True)
