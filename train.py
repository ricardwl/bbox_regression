import keras
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
import numpy as np
import pandas as pd
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint
def _run_in_batches(f, data_dict, out, batch_size,input_var):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        print(batch_data_dict[input_var].shape)
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def bboxreg_loss(y_true,y_pred):
    return K.sum(K.square(y_true-y_pred))
# def iou_loss(y_true, y_pred):
#     y_true = y_true[1:]
#     y_pred = y_pred[1:]
#     bbox_tl, bbox_br = y_true[:2], y_true[:2] + y_true[2:]
#     y_pred_tl = y_pred[ :2]
#     y_pred_br = y_pred[ :2] + y_pred[ 2:]
#
#     return area_intersection / (area_bbox + area_y_pred - area_intersection)
def iou_loss(y_true, y_pred):
    y_true = y_true[1:]
    y_pred = y_pred[1:]
    bbox_tl, bbox_br = y_true[:2], y_true[:2] + y_true[2:]
    y_pred_tl = y_pred[:, :2]
    y_pred_br = y_pred[:, :2] + y_pred[:, 2:]

    tl = np.c_[K.maximum(bbox_tl[0], y_pred_tl[ 0]),
               K.maximum(bbox_tl[1], y_pred_tl[ 1])]
    br = np.c_[K.minimum(bbox_br[0], y_pred_br[ 0]),
               K.minimum(bbox_br[1], y_pred_br[ 1])]
    wh = K.maximum(0., K.s)

    area_intersection = wh.prod(axis=1)
    area_bbox = y_true[2:].prod()
    area_y_pred = y_pred[2:].prod(axis=1)
    return area_intersection / (area_bbox + area_y_pred - area_intersection)

def exrtact_features(data_x, ckpt_name='model_data/mars-small128_new.pb', input_name='images',output_name='features',batch_size = 1):
    sess = tf.Session()

    with tf.gfile.GFile(ckpt_name,'rb') as file_handle:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file_handle.read())

    tf.import_graph_def(graph_def, name='net')
    input_var = tf.get_default_graph().get_tensor_by_name("net/%s:0" % input_name)
    output_var = tf.get_default_graph().get_tensor_by_name("net/%s:0" % output_name)
    feature_dim = output_var.get_shape().as_list()[-1]
    image_shape = input_var.get_shape().as_list()[1:]
    out = np.zeros((len(data_x), feature_dim), np.float32)

    _run_in_batches(
        lambda x: sess.run(output_var, feed_dict=x), {input_var: data_x}, out, batch_size,input_var)
    sess.close()

    return out

def gen_input(img_path):
    imgs = []
    img_patches = cv2.imread(img_path)
    img_patches = cv2.resize(img_patches, tuple([128, 64][::-1]))

    imgs.append(img_patches)
    imgs = np.asarray(imgs)
    return imgs


def build_bboxreg_net(
        input_shape = (1,128),
        loss=bboxreg_loss
):
    input = keras.layers.Input(shape=(input_shape))
    input2 = keras.layers.Input(shape=(1,5))
    dense1 = keras.layers.Dense(units=2018)(input)
    dense2 = keras.layers.Dense(units=2018)(dense1)
    dense3 = keras.layers.Dense(units=2018)(dense2)
    dense4 = keras.layers.Dense(units=5)(dense3)
    added = keras.layers.Add()([dense4, input2])
    model = keras.models.Model(inputs=[input,input2],output=added)
    model.compile(loss=loss, optimizer=Adam(1e-3, decay=1e-6),metrics=['mean_squared_error'])
    return model
def gen_y_train(gt_csv):
    y_train =[]
    for i in range(gt_csv.shape[0]):
        name = det_csv['id'][i]
        t,l,w,h,f =  int(gt_csv['t'][i]), int(gt_csv['l'][i]), int(gt_csv['w'][i]), int(gt_csv['h'][i]),int(gt_csv['flag'][i])

        y_train.append([t,l,w,h,f])
    return y_train


if __name__ == '__main__':

    model_filename = 'model_data/mars-small128_new.pb'
    det_csv = pd.read_csv('det.csv')
    gt_csv = pd.read_csv('gt.csv')
    print(det_csv.columns)
    x_train = []
    features = []
    for i in range(det_csv.shape[0]):
        name = det_csv['id'][i]
        t,l,w,h =  int(det_csv['t'][i]), int(det_csv['l'][i]), int(det_csv['w'][i]), int(det_csv['h'][i])
        img_path = '../data/overlaps/'+name+'.jpg'
        print(img_path)
        img_data = gen_input(img_path)
        features.append(exrtact_features(ckpt_name=model_filename,data_x=img_data))
        x_train.append([[t,l,w,h,0]])
    features_train = np.asarray(features)
    x_train = np.asarray(x_train)
    y_train = np.asarray(gen_y_train(gt_csv))
    print(x_train.shape)
    print(features_train)
    model = build_bboxreg_net(loss=iou_loss)
    checkpointer = ModelCheckpoint(  'bbox_best.h5', monitor='val_loss', verbose=1, save_best_only=True)
    model.fit(x=[features_train,x_train],y=y_train,batch_size=1,epochs=3000,validation_split=0.1,callbacks=[ checkpointer])

