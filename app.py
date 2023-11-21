import io
from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model('tu_modelo.h5')

# Nombres de las clases
class_names = ['Queratosis actínica', 'Carcinoma basocelular', 'Queratosis seborreica', 'Dermatofibroma', 'Melanoma', 'Nevo melanocítico', 'Hemangioma vascular']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener la imagen del formulario
        img = request.files['image'].read()

        # Procesar la imagen para que coincida con el formato esperado por el modelo
        img_array = image.img_to_array(image.load_img(io.BytesIO(img), target_size=(32, 32)))
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Realizar la predicción
        prediction = model.predict(img_array)

        # Obtener la clase predicha
        predicted_class = np.argmax(prediction, axis=1)

        # Obtener el nombre de la clase predicha
        predicted_class_name = class_names[int(predicted_class[0])]

        # Devolver el resultado como JSON
        result = {'class': predicted_class_name}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
