import keras
import numpy as np
import train
import keras.backend as K
from keras.optimizers import Adam
# def load_bboxmodel(model_path):
#     return load_model(model_path)
def bboxreg_loss(y_true,y_pred):
    return K.sum(K.square(y_true-y_pred))


def build_bboxreg_net(
        input_shape = (1,128),
        name_optimizer="sgd",
        rate_learning = 0.01,
        rate_decay_learning=0.0,
        output_dim = (1,5),
        name_initializer='glorot_normal'
):
    input = keras.layers.Input(shape=(input_shape))
    input2 = keras.layers.Input(shape=(0,5))
    dense1 = keras.layers.Dense(units=2018)(input)
    dense2 = keras.layers.Dense(units=2018)(dense1)
    dense3 = keras.layers.Dense(units=2018)(dense2)
    dense4 = keras.layers.Dense(units=5)(dense3)
    added = keras.layers.Add()([dense4, input2])
    model = keras.models.Model(inputs=[input,input2],output=added)
    model.compile(loss=bboxreg_loss, optimizer=Adam(1e-3, decay=1e-6),metrics=['mean_squared_error'])
    return model


if __name__ == '__main__':
    model_path = 'bbox_best.h5'
    model = build_bboxreg_net()
    model.load_weights(model_path)
    img_path = '../data/patches/80_70.jpg'
    img_data = train.gen_input(img_path)
    features = train.exrtact_features(data_x=img_data)
    features = np.asarray([features])
    x = np.asarray([[[622,440,88.0,292.5,0]]])
    print(model.predict([features,x]))