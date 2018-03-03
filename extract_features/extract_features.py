# -*- coding:utf-8 -*-
"""
Valing LightCNN.
"""
import argparse
import mxnet as mx
import cv2
import numpy as np
import scipy.io as sio

from gulon_lightcnn import LightCNN_29

def get_image(image_path):
    # download and show the image
    img = cv2.imread(image_path, 0) # gray
    if img is None:
        return None
    # convert into format (batch, RGB, width, height)
    # every image minus self mean. not the mean of all images.
    # img_1 = img_1 - [np.sum(img_1) / img_1.size]
    # every image minus the mean of all images. In mxnet, the RGB mean is '123.68,116.779,103.939',
    # so the gray mean is: 0.3 * 123.68 + 0.59 * 116.779 + 0.1 * 103.939 = 117.4396
    img = cv2.resize(img, (128, 128)) - [123.68] #[116.237197303]
    img = np.reshape(img, (1, 1, 128, 128))
    # The data setting of train phase and test phase must be same. 
    # If the training data scale by 1/255., then the testing data must scale by 1/255.
    # In mxnet, when loading data, the scale is 1., w.t. the image data range is [0, 255].

    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='../HybridBlock_LightCNN/model/LightCNN-10K-40.params',
                        help='The trained model to get feature')
    parser.add_argument('--num-classes', type=int, default=10000,
                        help='The class number of dataset')
    args = parser.parse_args()

    # Utlize gulon to load trained model.
    # fc1_output
    # fc1_net = LightCNN_29(num_classes=args.num_classes)
    # fc1_net.load_params(args.model_path, ctx=mx.gpu(0))

    # return mfm_fc1_output
    mfm_fc1_net = LightCNN_29(num_classes=args.num_classes)
    # Load trained parameters
    mfm_fc1_net.load_params(args.model_path, ctx=mx.gpu(0))

    # Dataset is lfw_patch, total have 13233 images. Extract every image's feture, the feature dim is 768/256.
    labels = np.empty([13233, 1], dtype=object)
    res = []
    count = 0
    with open("lfw_patch_part.txt", "r") as f:
        # lfw_patch_part.txt
        # 13233 samples.
        for line in f:
            name = []
            line = line.strip() # get rid of ' '!!
            name.append(line.split('/')[-2] + '/' + line.split('/')[-1])
            # print(name) # Aaron_Peirsol/Aaron_Peirsol_0004.jpg
            labels[count, 0] = name

            image_path = "/home/ly/DATASETS" + line

            img = get_image(image_path)
            # compute the predict probabilities
            input = mx.nd.array(img, ctx=mx.gpu(0)) # NDArray define in GPU.
            feature = mfm_fc1_net(input).asnumpy() # Transform to numpy array!! Do not forget!!
            # type(mfm_fc1_net(input)) is NDArray! Must transform to numpy ndarray!!
            feature = np.squeeze(feature)
            # print(feature.shape) # 768/256.
            res.append(feature)

            count += 1
            if count % 100 == 0:
                print("Images {}".format(count))

    res = np.array(res)
    res = np.reshape(res, [13233, 256]) # 768/256.
    np.save("lfw_maxout.npy", res)
    print (res.shape)
    print (labels.shape)
    sio.savemat("./LFW_features_maxout.mat", {'data':res, 'label':labels})
    f.close()
