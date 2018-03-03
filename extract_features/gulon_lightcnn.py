# -*- coding:utf-8 -*-
"""
Utlize gulon redefine the network. 
use net.hybridize()! Base class is nn.HybridBlock, and use nn.HybridSequential() construct a network.
In nn.HybridBlock, we should implement the hybrid_forward() method!
Symbol use the function, ndarray must can use! The name of function must be same!

NDArray operation can put in hybrid_forward() method! hybrid_forward() has another argument F. 
use F to determine use Symbol or NDArray. mnxet has Symbol API and NDArray API, the name of API is basic agreement.
Utilize F to access API, these API is not in thegulon, but in the mxnet.
F, mxnet.symbol or mxnet.ndarray.

Before net.hybridize(), F is using NDArray, atfer net.hybridize(), F is using Symbol.
Symbol code will not use python, but use C++ to compute!
"""
from mxnet.gluon import nn
# from mxnet import nd

# Every subnet define to a class. Inherited from nn.HybridBlock. NDArray flow!

class mfm(nn.HybridBlock):
    def __init__(self, num_filter, kernel_size, stride, padding, mfm_type, **kwargs):
        super(mfm, self).__init__(**kwargs)
        self.num_filter = num_filter
        self.mfm_type = mfm_type
        # mfm_type = 1, conv + ele-max + conv + ele-max; otherwise, conv + ele-max.
        # MFM is the element-wise max.
        # First, define all the needed operations. In class __init__, only define the mxnet.gluon.nn API.
        # There, operations only contain the mxnet.gluon.nn API.
        # do not include nd API.

        # Wrong!! Changing!
        # The layer with parameters must put in the __init__() of class, do not put in forward().
        # In fprward(), you can use ReLU, max, SliceChannel and so on, these operation have no parameters!!
        # The layer with parameters must be different, like: self.conv_op_1, self.conv_op_2, self.conv_op_3. 
        # Otherwise, the parameters will reuse!!
        # In forward(), the name of all operations with parameters should be different! 
        # Otherwise, the parameters will reuse!!
        # !! According to mfm_type, define conv2d, when mfm_type==1, define 2 conv2d, mfm_type==0, define only one
        # conv2d.
        # In here, self.conv_op_1 and self.conv_op_3 can be same actually.
        if self.mfm_type==1:
            self.conv_op_1 = nn.Conv2D(channels=self.num_filter, kernel_size=kernel_size, strides=stride, 
                padding=padding)
            self.conv_op_2 = nn.Conv2D(channels=self.num_filter, kernel_size=kernel_size, strides=stride, 
                padding=padding)
        else:
            self.conv_op_3 = nn.Conv2D(channels=self.num_filter, kernel_size=kernel_size, strides=stride, 
                padding=padding)
        # nn.Conv2D define in mxnet/gulon/nn/conv_layers.py.

    def hybrid_forward(self, F, x):
        # Because SliceChannel, maximum API in not in gulon, so these API must put in hybrid_forward() method.
        # mfm_type == 1 or mfm_type == 0.
        if self.mfm_type==1:
            # conv
            self.conv_1 = self.conv_op_1(x)

            # slice channel.
            self.slice_conv1 = F.SliceChannel(data=self.conv_1, num_outputs=3, axis=1)
            # SliceChannel, in hybrid_forward, it should use F.SliceChannel to implement the 
            # SliceChannel operate.
            # F.SliceChannel, slice NDArray/Symbol. argument is: data, num_outputs. 
            # return NDArray or list of NDArrays.
            # These API must input data.
            
            # element max, nd API. Symbol API and ndarray API is the same!
            self.aux_mfm_conv1 = F.maximum(self.slice_conv1[0], self.slice_conv1[1])
            # nd.maximum, argument is lhs, rhs. lhs and rhs is NDArray.
            self.mfm_conv1 = F.maximum(self.aux_mfm_conv1, self.slice_conv1[2])
            # conv
            self.conv = self.conv_op_2(self.mfm_conv1)
        else:
            self.conv = self.conv_op_3(x)

        # slice channel.
        self.slice_conv = F.SliceChannel(data=self.conv, num_outputs=3, axis=1)

        # element max, sym/nd API.
        self.aux_mfm_conv = F.maximum(self.slice_conv[0], self.slice_conv[1])
        # F.maximum, argument is lhs, rhs. lhs and rhs is NDArray.
        self.mfm_conv = F.maximum(self.aux_mfm_conv, self.slice_conv[2])

        return self.mfm_conv

class resblock(nn.HybridBlock):
    def __init__(self, num_blocks, num_filter, kernel_size, stride, padding, **kwargs):
        super(resblock, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        # mfm operation. define a object of class mfm.
        # this object will be used in hybrid_forward() method.

        # Wrong!!
        # The layer with parameters must put in the __init__() of class, do not put in forward().
        # In fprward(), you can use ReLU, max, SliceChannel and so on, these operation have no parameters!!
        # Redefine a sub net use nn.HybridSequential() or nn.Sequential().
        # The add of resblock_net is mfm_op, so we can use resblock_net[i](x) to do forward() operation.
        self.resblock_net = nn.HybridSequential()
        with self.name_scope():
            for i in range(self.num_blocks):
                self.resblock_net.add(
                    mfm(num_filter=num_filter, kernel_size=kernel_size, stride=stride, padding=padding, mfm_type=1))

    def hybrid_forward(self, F, x):
        '''
        residual blocks contain two 3x3 convolution layers, and two MFM operations without batch normalization.
        resblock contain num_blocks operations. num_blocks = [1, 2, 3, 4].
        a res block As follow:
            x
            |\
            | \
            |  conv2d + maximum
            |  conv2d + maximum
            | /
            |/
            + (addition here)
            |
           out
        repeat the block for num_blocks times.
        '''
        in_data = x
        # The add of resblock_net is mfm_op, so we can use resblock_net[i](x) to do forward() operation.
        for i, resblock_op in enumerate(self.resblock_net):
            self.mfm_x = resblock_op(in_data)
            out = self.mfm_x + in_data
            in_data = out

        return out # in_data = out

class LightCNN_29(nn.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(LightCNN_29, self).__init__(**kwargs)

        num_blocks = [1, 2, 3, 4]
        with self.name_scope():
            # conv net. include conv + element-max + pool.
            self.conv_net = nn.HybridSequential()
            self.conv_net.add(
                # mfm_1
                mfm(num_filter=144, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), mfm_type=0),
                # pool1
                nn.MaxPool2D(pool_size=2, strides=2),

                # mfm_2x
                resblock(num_blocks=num_blocks[0], num_filter=144, kernel_size=(3, 3), stride=(1, 1), 
                    padding=(1, 1)),
                # mfm_2a
                mfm(num_filter=144, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), mfm_type=0),
                # mfm_2
                mfm(num_filter=288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), mfm_type=0),
                # pool2
                nn.MaxPool2D(pool_size=2, strides=2),

                # mfm_3x
                resblock(num_blocks=num_blocks[1], num_filter=288, kernel_size=(3, 3), stride=(1, 1), 
                    padding=(1, 1)),
                # mfm_3a
                mfm(num_filter=288, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), mfm_type=0),
                # mfm_3
                mfm(num_filter=576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), mfm_type=0),
                # pool3
                nn.MaxPool2D(pool_size=2, strides=2),

                # mfm_4x
                resblock(num_blocks=num_blocks[2], num_filter=576, kernel_size=(3, 3), stride=(1, 1), 
                    padding=(1, 1)),
                # mfm_4a
                mfm(num_filter=576, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), mfm_type=0),
                # mfm_4
                mfm(num_filter=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), mfm_type=0),

                # mfm_5x
                resblock(num_blocks=num_blocks[3], num_filter=384, kernel_size=(3, 3), stride=(1, 1), 
                    padding=(1, 1)),
                # mfm_5a
                mfm(num_filter=384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), mfm_type=0),
                # mfm_5
                mfm(num_filter=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), mfm_type=0),
                # pool4
                nn.MaxPool2D(pool_size=2, strides=2),

                # flatten
                nn.Flatten(),
                # fc1
                nn.Dense(768) # default not use activation.            
                )
            
            '''
            # fc1
            self.fc1 = nn.HybridSequential()
            self.fc1.add(
                nn.Dense(768) # default not use activation.
                )
            '''

            # fc2
            self.fc2 = nn.HybridSequential()
            self.fc2.add(
                nn.Dense(num_classes) # default not use activation.
                )
            # It could write three sub net: conv_net + fc1 + fc2.

    def hybrid_forward(self, F, x):
        # conv net. include conv + element-max + pool.
        fc1_out = self.conv_net(x)

        # mfm_fc1, data must be NDArray.
        self.slice_fc1 = F.SliceChannel(data=fc1_out, num_outputs=3, axis=1)
        
        # element max, nd API.
        self.aux_mfm_fc1 = F.maximum(self.slice_fc1[0], self.slice_fc1[1])
        # F.maximum, argument is lhs, rhs. lhs and rhs is NDArray.
        self.mfm_fc1 = F.maximum(self.aux_mfm_fc1, self.slice_fc1[2])

        # fc2
        out = self.fc2(self.mfm_fc1)

        '''
        # Generate layer output. temporarily define a list, append fc1_output, mam_fc1_output, fc2_output.
        # The dict is a more good idea, the key is the layer name, the value is the layer output.
        # Just like net.collect_params(), it returns ParameterDict, the key is weight name, value is weight.
        net_output = []
        net_output.append(fc1_output)
        net_output.append(self.mfm_fc1)
        net_output.append(out)
        # In testing, it can not use net.hybridize(), so the data in the network is NDArray. In this way, we can
        # get somelayers output.
        # In forward(), something your return has no relation with the class object, when use object, w.t. a(x),
        # it will call forward() method.
        # So, the forward() can return whatever you want, before net.hybridize(), the data in the network is NDArray.
        return net_output
        '''

        return self.mfm_fc1

if __name__ == '__main__':
    net = LightCNN_29(10000)
    params = net.collect_params()
    print(params)