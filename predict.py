import numpy as np
# import keras as k

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model


longitud, altura = 100,100

modelo = './modelo/modelo.h5'
pesos_modelo = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura,3))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: Letra A")
  elif answer == 1:
    print("pred: Letra B")
  else:
    print("no reconocido")
  return answer
predict('WhatsApp Image 2019-08-10 at 11.30.02 AM.jpeg')

# result = load_img('dog.20.jpg', target_size=(longitud, altura, 3))
# x = img_to_array(result)
# x = np.expand_dims(x, axis=0)
# array = cnn.predict(x)
# result = array[0]
# answer = np.argmax(result)
# if answer == 0:
#   print("pred: Perro")
# elif answer == 1:
#   print("pred: Gato")
# elif answer == 2:
#   print("pred: Gorila")

