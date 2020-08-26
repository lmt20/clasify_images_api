from flask import Flask, jsonify, request
from app.Knn import train_test_modified


app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route("/test", methods=['GET'])
def get():
    similar_image_paths = train_test_modified.run("https://cafefcdn.com/thumb_w/650/2017/photo-1-1496895521714-crop-1496895541721-1496908353196.jpg")
    return jsonify({"SimilarImage" : similar_image_paths})

@app.route("/images", methods=['POST'])
def find_similar_images():
    try:
        req_data = request.get_json()
        image_path = req_data['urlPath']
        similar_image_paths = train_test_modified.run(image_path)
        return jsonify({"SimilarImages" : similar_image_paths})                                                                                                                         
    except :
        print("ERRR")
        pass
if __name__ == "__main__":
    app.run()
