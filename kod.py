# importy
import numpy as np
import matplotlib.pyplot as plot
from keras_preprocessing.image import load_img
from keras.models import load_model
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, image_utils
import cv2
import os

# przygotowanie modelu przedtreningowy

model_podstawowy = MobileNet(input_shape=(220, 220, 3), include_top=False)  # wagi
for warstwa in model_podstawowy.layers:
    warstwa.trainable = False

X = Flatten()(model_podstawowy.output)
X = Dense(units=3, activation='softmax')(X)

# tworzenie modelu do trenowania danych

modelG = Model(model_podstawowy.input, X)

modelG.summary()

modelG.compile(optimizer="adam", loss=categorical_crossentropy, metrics=['accuracy'])

# przygotowanie danych do użycia generatora danych

genData_train = ImageDataGenerator(
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=False,
    rescale=1. / 255
)

data_train = genData_train.flow_from_directory(
    "train2",
    target_size=(220, 220),
    batch_size=2,
)

print('Znalezione kategorie: ', data_train.class_indices)


dataG_val = ImageDataGenerator(rescale=1 / 255)

# dane przeznaczone do walidacjii

data_val = dataG_val.flow_from_directory(directory="train2",
                                         target_size=(220, 220),
                                         batch_size=2,
                                         )
print(data_val.class_indices)

# tworzymy wizualizację zdjęć w treningowym generatorze danych

zd_img, label = data_train.next()


# funkcja chroniąca zdjęcia po przetworzeniu

def plotZdjecia(img_arr, label):
    licznik = 0
    for im, l in zip(img_arr, label):
        plot.imshow(im)
        plot.title(im.shape)
        plot.axis = False
        plot.show()

        licznik += 1
        if licznik == 10:
            break


# wywołanie funckcji plotZdjecia()

plotZdjecia(zd_img, label)

# --------------------------

# tworzenie kontroli modelu i wczesnego zatrzymywanie (early stopping)

eStop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=5, verbose=1, mode='auto')

# check point
checkModel = ModelCheckpoint(filepath="./best_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                             mode='auto')
checkModel
backCall = [eStop, checkModel]

# trenowanie modelu
hist = modelG.fit(data_train,
                  steps_per_epoch=len(data_train),
                  epochs=30,
                  validation_data=data_val,
                  validation_steps=len(data_val),
                  callbacks=[eStop, checkModel])

modelG = load_model("./best_model.h5")

h = hist.history

# podsumowanie na wykresie przebiegu trenowania modelu dla dokładności dopasowania

#plot.plot(h['accuracy'])
#plot.plot(h['val_accuracy'])
#plot.title("model accuracy")
#plot.ylabel('accuracy')
#plot.xlabel('epoch')
#plot.legend(['train', 'test'], loc='upper left')
#plot.show()

# podsumowanie na wykresie przebiegu trenowania modelu dla strat w dopasowaniu

#plot.plot(h['loss'])
#plot.plot(h['val_loss'])
#plot.title("model loss")
#plot.ylabel('loss')
#plot.xlabel('epoch')
#plot.legend(['train', 'test'], loc='upper left')
#plot.show()

numPorz = data_train.class_indices.values()
nazwy_emocji = data_train.class_indices.keys()
op = dict(zip(numPorz, nazwy_emocji))

# testowanie algorytmu

# wczytywanie zdjęcia testowego



#zdj = load_img(sciezka, target_size=(500, 500))

folder_dir = 'test'
imgs = []

for images in os.listdir(folder_dir):

    zdj = cv2.imread(os.path.join(folder_dir, images))
    width = 220
    height = 220
    dsize = (width, height)

    zdj = cv2.resize(zdj, dsize)

# konwersja zdjęcia jako tablica

    i = image_utils.img_to_array(zdj)
    i = i/255
    input_arr = np.array([i])
    input_arr.shape

# " odgadywanie rodzaju zdjęcia"
    pred = np.argmax(modelG.predict(input_arr))

# wyswietlenie zdjęcia z podpisem
    title1 = f" Zdjecie jest {op[pred]}"
    print(title1)
    plot.imshow(input_arr[0])
    plot.title(title1)
    plot.show()
