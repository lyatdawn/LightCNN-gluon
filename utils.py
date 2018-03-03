# -*- coding:utf-8 -*-
"""
Utlize codes.
"""
import os
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
import mxnet as mx
from time import time
import logging

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])

def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            acc += nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n

def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, lr_step_epochs=None, lr_decay=0.1,
        print_batches=100, load_epoch=0, model_prefix=None, period=1):
    """
    Train a network.
    required=True for those uninitialized arguments.
    Refer to mxnet/module/base_module.py fit() to load trained model.
    Refer to fit.py to set lr scheduler.
    """
    logging.info("Start training on {}".format(ctx))
    # Load trained model.
    # Indicates the starting epoch. Usually, if resumed from a checkpoint saved at a previous training phase 
    # at epoch N, then this value is N.
    if load_epoch > 0:
        if os.path.exists(model_prefix + "-{}.params".format(load_epoch)):
            net.load_params(model_prefix + "-{}.params".format(load_epoch), ctx)
            logging.info("Resume training from epoch {}".format(load_epoch))
        else:
            print("The resume model does not exist.")

    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    for epoch in range(load_epoch, num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        # Set lr scheduler.
        # can use learning_rate() and set_learning_rate(lr) to set lr scheduler.
        if lr_step_epochs is not None:
            step_epochs = [int(l) for l in lr_step_epochs.split(',')]
            for s in step_epochs:
                if epoch == s:
                    trainer.set_learning_rate(trainer.learning_rate * lr_decay)
                    logging.info("Adjust learning rate to {} for epoch {}".format(trainer.learning_rate, epoch))
                    # Use trainer.learning_rate

        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        start = time()
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
                              for yhat, y in zip(outputs, label)])
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in label])
            if print_batches and (i+1) % print_batches == 0:
                logging.info("Epoch [%d]. Batch [%d]. Loss: %f, Train acc %f" % 
                    (epoch, n, train_loss/n, train_acc/m))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        logging.info("Epoch [%d]. Loss: %f, Train acc %f, Test acc %f, Time %f sec" % (
            epoch, train_loss/n, train_acc/m, test_acc, time() - start))
        # Time, one epoch spend time. include train and test.

        # save checkpoint
        if (epoch + 1) % period == 0:
            net.save_params(model_prefix + "-{}.params".format(epoch + 1))
            logging.info("Saved checkpoint to {}-{}.params".format(model_prefix, epoch + 1))
