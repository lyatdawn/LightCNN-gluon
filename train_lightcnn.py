# -*- coding:utf-8 -*-
"""
Train and Validate LightCNN-29.
Utlize some codes inb utils.py to train/val self model.
"""
import os
import argparse
import logging
import mxnet as mx
from mxnet import gluon
from gulon_lightcnn import LightCNN_29
import data
import utils

# logging
log_file = "./model/LightCNN-10K.log"
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                   filename=log_file,
                   level=logging.INFO,
                   filemode='a+')
logging.getLogger().addHandler(logging.StreamHandler())
# logging.info

# data load method two. use common/data.py dataget_rec_iter() to load ImageRecordIter.
def load_rec_iter():
    data_dir = "/home/ly/DATASETS/MsCelebV1/MXNet_MsCeleb_10K" # MsCeleb 10K
    # REC data is 144*144 image, use random crop to 128*128.
    fnames = (os.path.join(data_dir, "MsCeleb_train.rec"), os.path.join(data_dir, "MsCeleb_val.rec")) # MsCeleb 10K

    return fnames

if __name__ == '__main__':
	# load rec data
    (train_fname, val_fname) = load_rec_iter()

    # parse args
    parser = argparse.ArgumentParser(description="train LightCNN-9 and LightCNN-29.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    parser.set_defaults(
        # data
        num_classes = 10000, # MsCeleb 10K
        # data augmentations
        data_train = train_fname,
        data_val = val_fname,
        image_shape = '1,128,128', # image shape
        pad_size = 4,
        data_nthreads = 8, # number of threads for data decoding
        # train
        kv_store = 'device',
        disp_batches = 100,
        batch_size = 64,
        num_epochs = 35, # 35, epoch from 0 to 34.
        # optim
        # From random init to train.
        optimizer = 'adam', 
        lr = 0.00001, # init lr is 1e-5.
        # lr_step_epochs, the epochs to reduce the lr, e.g. lr_step_epochs = '15,20'.
        # When the epoch >= 27(epoch begin from 0), the train loss decrease slowly, so we can use reduce the 
        # learning rate to increase the accuracy. We might use lr_step_epochs earlier, e.g. epoch 20, 22. 
        lr_step_epochs = '26,28,30,32,34',
        # lr_decay, the ratio to reduce lr on each step. e.g. lr_decay = 0.1.
        lr_decay = 0.1,

        # chechpoint
        # load_epoch. Load trained model, load_epoch is the epoch of the model. e.g. load_epoch = 28.
        load_epoch = 0, # Load trained model. if load_epoch is 0, represent from random init to train.
        # model_prefix, the prefix of save checkp, e.g., LightCNN-10K-1.params.
        model_prefix = 'model/LightCNN-10K',
    )
    args = parser.parse_args()
    # context
    ctx = utils.try_gpu()

    # network
    net = LightCNN_29(num_classes=args.num_classes)

    # init weight and bias
    # init. define init method. 1) define a custom init class; 2) Utlize Parameter.set_data change the bias directly.
    # refer to mxnet/initializer.py, add a class Xavier_LightCNN, the bias is inited to 0.1, not 0.
    # The Xavier_LightCNN class is for LightCNN model. In class Xavier_LightCNN, change bias init method.
    net.initialize(ctx=ctx,
        init=mx.initializer.Xavier_LightCNN(rnd_type="uniform", factor_type="avg", magnitude=1.0))
    # initialize() define in mxnet/gulon/parameter.py.

    # loss
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # data. still use data.py to load data.
    (train, val) = data.get_rec_iter(args, kv=mx.kvstore.create(args.kv_store))

    # train and val
    net.collect_params().reset_ctx(ctx)
    net.hybridize() # Before net.hybridize(), F is using NDArray, atfer net.hybridize(), F is using Symbol.
    # Symbol code will not use python, but use C++ to compute!

    # Refer to http://zh.gluon.ai/chapter_computer-vision/kaggle-gluon-cifar10.html, set lr scheduler.
    # The trainer is defined in the main function. The lr scheduler is used in utils.py.
    '''
    In mxnet/gulon/trainer.py, the class Trainer has these method:
    1) learning_rate(), return current learning rate.
    2) set_learning_rate(lr), set the learning rate will be used.
    3) step(batch_size, ignore_stale_grad=False), Makes one step of parameter update based on batch_size data.
    4) save_states(fname), Saves trainer states (e.g. optimizer, momentum) to a file.
    5) load_states(fname), Loads trainer states (e.g. optimizer, momentum) from a file.

    So, we can use learning_rate() and set_learning_rate(lr) set lr scheduler.
    use trainer.learning_rate!
    '''
    trainer = gluon.Trainer(net.collect_params(),
              args.optimizer, {'learning_rate': args.lr})

    utils.train(train_data=train, 
                test_data=val, 
                net=net, 
                loss=loss,
                trainer=trainer,
                ctx=ctx,
                num_epochs=args.num_epochs,
                lr_step_epochs=args.lr_step_epochs,
                print_batches=args.disp_batches,
                load_epoch=args.load_epoch,
    	        model_prefix=args.model_prefix)

