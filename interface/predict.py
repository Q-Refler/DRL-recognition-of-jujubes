import numpy as np
import tensorflow as tf

import os

save_path = os.path.join(os.path.dirname(__file__) + '/', '../exe_no_ui\DenseNet121\checkpoint_model')
print(save_path)
model = tf.keras.models.load_model(save_path)
model.summary()


#  检测函数
def AI_predict(zao_img):
    x_predict = zao_img[tf.newaxis, ...]
    result = model.predict(x_predict)
    pred = np.argmax(result, axis=1)

    return pred
