import sys      #Sirve para moverse entre carpetas del SO
import os       #Sirve para moverse entre carpetas del SO
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator      #Ayuda a preprocesar las imagenes para entrenar los algoritmos
from tensorflow.python.keras import optimizers                                  #Optimizador con el cual se va a entrenar nuestro algoritmo
from tensorflow.python.keras.models import Sequential                           #Permite hacer redes neuronales secuenciales (cada una de las capas esta en orden)
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation  #Para poner en una dimension toda la informacion
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D          #Son las capas en las cuales vamos a hacer la convulsiones y maxpooling
from tensorflow.python.keras import backend as K                                #Mata las sesiones de keras que se haya ejecutado

#Limpiamos las sesiones activas
K.clear_session()

#DECLARACION DE VARIABLES
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

#PARAMETROS
epocas = 50    #Numero de veces que se va a iterar sobre el set de datos durante el entrenamiento
altura, longitud = 100, 100     #Tamaño de procesamiento de las imagenes
batch_size = 20     #Numero de imagenes que vamos a procesar en cada uno de los pasos
pasos = 1000 #Numero de veces que se va a procesar la informacion en cada una de las epocas
pasos_validacion = 100   #Al final de cada epoca se va a correr 200 pasos con el set de datos de validacionphyton 
filtrosConv1 = 32   #Despues de aplicar la convolucion la imagen va tener una profundidad de 32
filtrosConv2 = 64   #Despues de aplicar la convolucion la imagen va tener una profundidad de 64
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_pool = (2,2)
clases = 2
lr = 0.0005  #Que tan grande seran los ajustes de la red para acercarse a la solucion optima

##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

test_datagen = ImageDataGenerator(
    rescale=1. / 255
    )

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

cnn = Sequential()
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.2))
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=pasos_validacion)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')