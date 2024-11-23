from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, make_response
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model = keras.models.load_model("model.h5")

@app.route("/upload", methods=["POST"])
def upload():
    
    if "image" not in request.files:
        return jsonify({"error": "Missing required request"})
    
    image_file = request.files["image"]
    image_file.save("img.jpg")
    path = "img.jpg"

    try:
        class_indices = {
            0: 'Canang',
            1: 'Gebogan',
            2: 'Pelangkiran',
            3: 'Pelinggih',
            4: 'Penjor',
            5: 'Banten Saiban',
            6: 'kain poleng',
            7: 'Tidak Terdeteksi'
        }

        img = image.load_img(path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=20)
        print("Predicted class:", classes[0])
        # Add the following line to print the perspective class
        perspective_class = np.argmax(classes[0])
        print("Perspective class:", class_indices[perspective_class])

        os.remove(path)
        return jsonify({"predicted_class": str(perspective_class), "name": class_indices[perspective_class]})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))