# 导入相应的库
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
from datetime import datetime
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import back.p46_cifar10_resnet18

print("TensorFlow version: ", tf.__version__)

# 设置图片的高和宽，一次训练所选取的样本数，迭代次数
im_height = 224
im_width = 224
batch_size = 16
epochs = 40 
versions = ['inceptionv3_new', 'Resnet18', 'mobelnetv2', 'InceptionResNetV2', 'DenseNet121_old_data']
version = 0  # 版本
image_path = "E:/Files/dataset/set_2"  # 数据集路径

train_dir = image_path + '/' + "train"  # 训练集路径
validation_dir = image_path + '/' + "test"  # 验证集路径

# 定义训练集图像生成器，并进行图像增强
train_image_generator = ImageDataGenerator(rescale=1. / 255,  # 归一化
                                           rotation_range=40,  # 旋转范围
                                           width_shift_range=0.2,  # 水平平移范围
                                           height_shift_range=0.2,  # 垂直平移范围
                                           shear_range=0.2,  # 剪切变换的程度
                                           zoom_range=0.2,  # 剪切变换的程度
                                           horizontal_flip=True,  # 水平翻转
                                           fill_mode='nearest')

# 使用图像生成器从文件夹train_dir中读取样本，对标签进行one-hot编码
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,  # 从训练集路径读取图片
                                                           batch_size=batch_size,  # 一次训练所选取的样本数
                                                           shuffle=True,  # 打乱标签
                                                           target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                           class_mode='categorical')  # one-hot编码

# 训练集样本数
total_train = train_data_gen.n

# 定义验证集图像生成器，并对图像进行预处理
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # 归一化

# 使用图像生成器从验证集validation_dir中读取样本
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,  # 从验证集路径读取图片
                                                              batch_size=batch_size,  # 一次训练所选取的样本数
                                                              shuffle=False,  # 不打乱标签
                                                              target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                              class_mode='categorical')  # one-hot编码

# 验证集样本数
total_val = val_data_gen.n

# 构建模型

# covn_base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, include_top=False,
#                                               weights='imagenet')

savepath = 'E:/Files/pycode/Net_train/model/' + versions[version] + '/'
checkpoint_save_path = savepath + 'checkpoint_model/'
if os.path.exists(checkpoint_save_path + '/variables/variables.index'):
    print('-------------load the model -----------------')
    model = tf.keras.models.load_model(checkpoint_save_path)
    model.summary()  # 打印每层参数信息
else:
    # 构建模型
    print('-------------build the model -----------------')
    # model = tf.keras.Sequential()
    # model.add(covn_base)
    # model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
    # model.add(tf.keras.layers.Dense(4, activation='softmax'))  # 加入输出层(4分类)
    # covn_base = tf.keras.applications.MobileNetV3Small(
    #     input_shape=(224, 224, 3), alpha=1.0, include_top=False,
    #     weights='imagenet')

    # covn_base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, include_top=False,
    #
    #                                              weights='imagenet')
    model = tf.keras.Sequential()
    if version == 0:  # InceptionV3
        covn_base = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model.add(covn_base)
        model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
        model.add(tf.keras.layers.Dense(4, activation='softmax'))  # 加入输出层(4分类)
        model.summary()
    elif version == 1:  # Resnet18
        model = back.p46_cifar10_resnet18.ResNet18([2, 2, 2, 2])
    elif version == 2:  # mobilenetV2
        covn_base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, include_top=False,
                                                      weights='imagenet')
        model.add(covn_base)
        model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
        model.add(tf.keras.layers.Dense(4, activation='softmax'))  # 加入输出层(4分类)
        model.summary()
    elif version == 3:  # InceptionResNetV2
        covn_base = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',
                                                                                input_shape=(224, 224, 3))
        for layer in covn_base.layers[:-30]:
            layer.trainable = False
        model.add(covn_base)
        model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
        model.add(tf.keras.layers.Dense(4, activation='softmax'))  # 加入输出层(4分类)
        model.summary()
    elif version == 4:  # DenseNet121
        covn_base = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet',
                                                               input_shape=(224, 224, 3))
        # for layer in covn_base.layers[:-30]:
        #     layer.trainable = False
        model.add(covn_base)
        model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
        model.add(tf.keras.layers.Dense(4, activation='softmax'))  # 加入输出层(4分类)
        model.summary()
    else:
        print('wrong version')
        quit()
    # 编译模型
    print('-------------complie the model -----------------')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 使用adam优化器，学习率为0.00001
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 交叉熵损失函数
                  metrics=["categorical_accuracy"])  # 评价函数

# checkpoint 回调 用于记录模型权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=False,
                                                 save_best_only=True,
                                                 )

# tensorbord 回调 用于记录训练过程数据
logdir = savepath + 'logs/scalars/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()


# 衰减学习率回调
def scheduler(epoch, lr):
    x = lr * tf.math.exp(-0.05)
    tf.summary.scalar('learning rate', data=x, step=epoch)
    return x


learn_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
print('-------------test information-----------------')
print(tf.config.list_physical_devices('GPU'))
print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
print(tf.test.is_built_with_cuda())
print('-------------fit the model -----------------')
history = model.fit(x=train_data_gen,  # 输入训练集
                    steps_per_epoch=total_train // batch_size,  # 一个epoch包含的训练步数
                    epochs=epochs,  # 训练模型迭代次数
                    verbose=1,
                    validation_data=val_data_gen,  # 输入验证集
                    validation_steps=total_val // batch_size,  # 一个epoch包含的训练步数
                    callbacks=[cp_callback,
                               learn_callback,
                               tensorboard_callback]
                    )

model.summary()  # 打印每层参数信息
