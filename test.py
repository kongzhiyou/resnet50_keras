from keras.applications import ResNet50
from keras.layers import Dense
from keras.engine import Model
from keras.optimizers import Adam
from keras.models import load_model
import glob
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from class_list import class_list
import os

'''
根据字典的值进行排序
'''
def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort(reverse=True)
    return [ backitems[i][1] for i in range(0,len(backitems))]

'''
模型定义并加载
'''
def finetune():
    pretrain_model = ResNet50(include_top=False,
                              input_shape=(224, 224, 3),
                              pooling='avg')
    '''
    加入模型输出层（倒数第二层加了128个全连接层，输出层是78个全连接层（按照分类数指定））
    '''
    x = pretrain_model.output
    #x = keras.layers.Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(78, activation='softmax', name='fc')(x)
    model = Model(input=pretrain_model.input, output=x)
    '''
    优化器
    '''
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model = load_model('third.32-0.98.h5')

img_path = glob.glob('/Users/peter/data/jshl/test/*')
for path in img_path:
    image_list = glob.glob(path + '/*.jp*g')
    for image in image_list:
        class_pre = {}
        img = Image.open(image).convert('RGB') # 转成3通道,和输入对应
        validation_batch = np.stack([preprocess_input(np.array(img.resize((224, 224))))])
        pred_probs = model.predict(validation_batch) # 一张图片在所有分类预测的置信度
        #每个分类对应的置信度输入到一个dict里
        for i in range(len(pred_probs[0])):
            class_pre.update({class_list[i]:pred_probs[0][i]})
        #根据置信度的值，对字典进行递减排序
        value = sort_by_value(class_pre)
        res1 = np.argmax(pred_probs[0]) # 从置信度中取最大的值
        prob = [value[0],value[1],value[2]] #获取置信度最大的Top3
        print(path, class_list[res1]) # 预测结果



