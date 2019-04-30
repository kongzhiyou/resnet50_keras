import cv2
import glob
from PIL import Image
from PIL import ImageFilter
import os

stride = [0.99,0.98,0.97]
root_path = '/Users/peter/data/jshl/train'
image_path = glob.glob(root_path+'/*')


'''
裁剪
'''
def image_crop():
    for path in image_path:
        image_list = glob.glob(path+'/*')
        if(len(image_list)<50):
            for image in image_list:
                img = cv2.imread(image)
                y,x = img.shape[0:2]
                for i in range(0,len(stride)):
                    cropped = img[0:int(y*stride[i]),0:int(x*stride[i])]  # (left, upper, right, lower)
                    image_name = image.split('/')[-1].split('.')[0]
                    suffix_name = image.split('/')[-1].split('.')[1]
                    img_path = path+"/"+image_name+'_aug_'+str(stride[i])+'.jpg'
                    try:
                        cv2.imwrite(img_path,cropped)
                    except Exception:
                        print('文件出错')

'''
图像滤波
'''
def image_filters(image):
    img = Image.open(image)
    img = img.convert('RGB')
    '''高斯滤波'''
    im_gblur = img.filter(ImageFilter.GaussianBlur)
    im_unsharp = img.filter(ImageFilter.UnsharpMask)
    im_blur = img.filter(ImageFilter.BLUR)
    '''细节增强滤波'''
    im_detail = img.filter(ImageFilter.DETAIL)
    '''边缘增强'''
    im_edge_enhance = img.filter(ImageFilter.EDGE_ENHANCE)
    '''平滑滤波'''
    im_smooth = img.filter(ImageFilter.SMOOTH)

    save_image(im_gblur, image, 'gaussianblur')
    save_image(im_unsharp, image, 'unsharpmask')
    save_image(im_blur,image,'blur')
    save_image(im_detail,image,'detail')
    save_image(im_edge_enhance,image,'endge_enhance')
    save_image(im_smooth,image,'smooth')

'''
抽取通道
'''
def rgb_convert(image):
    img = Image.open(image)
    img = img.convert('RGB')
    r,g,b = img.split()
    r = r.convert("RGB")
    g = g.convert("RGB")
    b = b.convert("RGB")

    save_image(r,image,'r')
    save_image(g, image,'g')
    save_image(b, image,'b')

'''
保存图片
'''
def save_image(img,image,op):
    new_name = image.split('/')[-1].split('.')[0] + '_aug_'+op+'_'+ '.jpg'
    '''得到父目录'''
    cag = os.path.abspath(os.path.dirname(image)+os.path.sep+'.')
    save_path = cag + '/' + new_name
    img.save(save_path)

if __name__ == '__main__':
    # for cag in image_path:
    #     image_list = glob.glob(cag+'/*')
    #     for image in image_list:
    #         RGB_convert(cag,image)
    image_filters('/Users/peter/data/sinho/sinho_others/-2871928887068713438.jpg')

