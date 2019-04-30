import os
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import ResNet50
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from loss_Impl import focal_loss

'''
模型定义
'''
def finetune(train_generator, valid_generator):
    pretrain_model = ResNet50(include_top=False,
                              input_shape=(224, 224, 3),
                              pooling='avg')
    '''
    加入模型输出层（倒数第二层加了128个全连接层，输出层是78个全连接层（按照分类数指定））
    '''
    x = pretrain_model.output
    #x = keras.layers.Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(79, activation='softmax', name='fc')(x)
    model = Model(input=pretrain_model.input, output=x)
    '''
    优化器
    '''
    optimizer = Adam(lr=0.001)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  使用交叉熵loss
    model.compile(optimizer=optimizer,loss=focal_loss(alpha=1.),metrics=['accuracy'])

    '''
    模型保存设置
    '''
    ckpt = ModelCheckpoint(filepath='./third.{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', save_best_only=False)
    es = EarlyStopping(monitor='val_loss', patience=5)
    '''
    Tensorboard训练过程可视化
    '''
    tb_cb = TensorBoard(log_dir='', write_images=1, histogram_freq=1)
    '''
    指定每隔多少个epoch学习率下降一次，并设置下降因子，例如factor=0.1,每次下降0.1
    '''
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', factor=0.3)
    '''
    模型保存指定路径
    '''
    get_lr = LambdaCallback(on_epoch_begin=lambda epoch, logs:print('epoch:{}\t learning rate:{}'.format(epoch, logs['lr'])))
    '''
    做数据增强并分批次训练，训练集是在不断增强的
    '''
    model.fit_generator(train_generator, steps_per_epoch=43000//32, epochs=30, callbacks=[ckpt, reduce_lr], validation_data=valid_generator, validation_steps=390/32)

    #model.fit() 不做数据增强，分批次训练，训练集是整个样本集
    model.save('xinhe_resnet_finetune_last_conv_1.h5')

'''
指定GPU
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
图像增强
'''
train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    # '/Users/peter/data/sinho/train',
    '/home/pinshi/jml/train',
    batch_size=32,
    class_mode='categorical',
    target_size=(224, 224))


validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    # '/Users/peter/data/sinho/validation',
    '/home/pinshi/jml/validation',
    shuffle=False,
    class_mode='categorical',
    target_size=(224,224))

finetune(train_generator,validation_generator)


