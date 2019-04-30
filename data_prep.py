import numpy as np
from PIL import Image
import os
import glob
from keras.applications.imagenet_utils import preprocess_input
import pickle

'''
处理分类数据将分类信息转换成"独热码"
'''
def one_hot(cls, cls_num):
    vec = np.zeros((cls_num,), dtype=np.uint)
    vec[cls] = 1
    return vec

'''
image：测试或训练图片
'''
def process_image(image,image_path):
    img = image.load_img(image_path, target_size=(224,224))
    x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

global img_arr_random
global annotations_random



def data_generator(data_pickle_file, dataset,  batch_size=32):
    if not os.path.exists(data_pickle_file):
        classes = sorted([d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))])
        print(classes)
        images1 = glob.glob(os.path.join(dataset, '*/*.jp*g')) #
        images2 = glob.glob(os.path.join(dataset, '*/*.png')) #
        images = images1 + images2
        img_arr = np.array([process_image(im) for im in images]    )
        annotations = np.array([one_hot(classes.index(i.split('/')[-2]),len(classes)) for i in images])
#        pickle.dump((img_arr, annotations), open(data_pickle_file, 'wb'))
        print('classes:{}\t images:{}\t img_arr shape:{}\t annotations shape:{}'.format(len(classes), len(images), img_arr.shape, annotations.shape))
    else:
        img_arr, annotations = pickle.load(open(data_pickle_file, 'rb'))
        print('img_arr shape:{}\t annotations shape:{}'.format(img_arr.shape, annotations.shape))

    size = len(annotations)
    random_indx = np.random.permutation(size)
    global img_arr_random
    img_arr_random = img_arr[random_indx]
    print('img_arr_random address:', id(img_arr_random))
    global annotations_random
    annotations_random = annotations[random_indx]
    while True:
        for i in range(size//batch_size+1):
            yield (img_arr_random[i*batch_size:(i+1)*batch_size], annotations_random[i*batch_size:(i+1)*batch_size])



