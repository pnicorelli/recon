import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


from keras.applications.resnet50 import ResNet50
modelRN50 = ResNet50(weights='imagenet')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

def RN50(image):
    preds = model_predict(image, modelRN50)
    pred_class = decode_predictions(preds, top=5)
    return pred_class[0]

##########################
##########################
# MODEL_PATH = 'pretrained_models/your_model.h5'
# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
