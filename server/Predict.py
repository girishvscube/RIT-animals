

animal_info = [
    {
        "name": "Cow",
        "scientific_name": "Bos taurus",
        "recognition": "Cows are recognized within a few hours of birth due to their distinctive appearance and behavior.",
        "discovery_year": "Neolithic era (Domestication)",
        "first_identified": "Domestication traces back to around 10,000 years ago."
    },
    {
        "name": "Dog",
        "scientific_name": "Canis lupus familiaris",
        "recognition": "Dogs are recognized shortly after birth, usually within a few weeks, when they start displaying distinct physical and behavioral traits.",
        "discovery_year": "Prehistoric times",
        "first_identified": "Dogs have been associated with humans for thousands of years, potentially dating back over 30,000 years."
    },
    {
        "name": "Donkey",
        "scientific_name": "Equus africanus asinus",
        "recognition": "Donkeys are recognized soon after birth, within hours to a few days, by their distinctive appearance and behavior.",
        "discovery_year": "Domesticated around 4000 BC",
        "first_identified": "Domestication occurred around the fourth millennium BC."
    },
    {
        "name": "Elephant",
        "scientific_name": "Elephas maximus (Asian Elephant) / Loxodonta africana (African Elephant)",
        "recognition": "Elephants are recognized at birth due to their size, appearance, and close family bonding behavior.",
        "discovery_year": "Ancient times",
        "first_identified": "Known and depicted in ancient civilizations like the Indus Valley and Egyptian cultures."
    },
    {
        "name": "Gorilla",
        "scientific_name": "Gorilla gorilla (Western Gorilla) / Gorilla beringei (Eastern Gorilla)",
        "recognition": "Gorillas are recognized shortly after birth, usually within a few hours, due to their distinctive physical appearance and close family ties.",
        "discovery_year": "19th century",
        "first_identified": "First scientifically described by Europeans in the 19th century."
    },
    {
        "name": "Horse",
        "scientific_name": "Equus ferus caballus",
        "recognition": "Horses are recognized within hours after birth due to their distinct appearance and ability to stand and walk shortly after being born.",
        "discovery_year": "Ancient times",
        "first_identified": "Horses have been domesticated for several thousand years."
    },
    {
        "name": "Lion",
        "scientific_name": "Panthera leo",
        "recognition": "Lions are recognized shortly after birth, usually within a few days, due to their distinctive appearance and bonding with the mother.",
        "discovery_year": "Ancient times",
        "first_identified": "Known and depicted in ancient civilizations."
    },
    {
        "name": "Owl",
        "scientific_name": "Order: Strigiformes (Various species)",
        "recognition": "Owls are recognized within a few weeks after hatching, when they start to develop their distinctive feathers and behaviors.",
        "discovery_year": "Varies by species",
        "first_identified": "Records of owl depictions date back to ancient cultures."
    },
    {
        "name": "Whale",
        "scientific_name": "Various species across families like Balaenopteridae, Physeteridae, etc.",
        "recognition": "Whales are recognized shortly after birth, usually within a few hours, due to their size and appearance alongside the mother.",
        "discovery_year": "Ancient times",
        "first_identified": "Whales have been hunted and depicted in ancient cultures."
    },
    {
        "name": "Zebra",
        "scientific_name": "Various species like Equus zebra, Equus grevyi, etc.",
        "recognition": "Zebras are recognized soon after birth, within an hour or so, due to their unique stripes and the bonding with the mother.",
        "discovery_year": "Ancient times",
        "first_identified": "Known and depicted in ancient cultures."
    }
]



# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# app = Flask(__name__)
# # Load the saved model
# model = tf.keras.models.load_model('animal_classification_model.h5')



# img_width, img_height = 150, 150  # Same dimensions as during training
# image_path = './cow.jpg'  # Path to the image you want to predict

# # Load the image and preprocess it for prediction
# img = image.load_img(image_path, target_size=(img_width, img_height))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.0  # Normalize the image data


# prediction = model.predict(img_array)



# # Get the class labels (assuming you have access to train_generator)
# # class_labels = train_generator.class_indices



# predicted_label = animal_info[np.argmax(prediction)]

# # Convert the prediction to a readable label
# print([np.argmax(prediction)])
# print(predicted_label)


# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image as PIL_Image 
from flask_cors import CORS

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes

# Load the saved model
model = tf.keras.models.load_model('animal_classification_model.h5')

img_width, img_height = 150, 150  # Same dimensions as during training

@app.route('/predict_animal', methods=['POST'])
def predict_animal():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    img = PIL_Image.open(image_file)
    img = img.resize((img_width, img_height))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image data

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_animal_info = animal_info[predicted_index]

    return jsonify({
        'name': predicted_animal_info['name'],
        'scientific_name': predicted_animal_info['scientific_name'],
        'recognition': predicted_animal_info['recognition'],
        'discovery_year': predicted_animal_info['discovery_year'],
        'first_identified': predicted_animal_info['first_identified']
    })



if __name__ == '__main__':
    app.run(debug=True)
