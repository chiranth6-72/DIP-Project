from flask import Flask, request, send_file
import code_module

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image = request.files['image']
    

    # Return the processed image as a response
    return send_file(image, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
