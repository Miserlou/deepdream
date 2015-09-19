# the dir frames need to be created!
# branch the google project

import sys, os

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe

import os

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist


def create_net(model_file):
    net_fn = os.path.join(os.path.split(model_file)[0], 'deploy.prototxt')
    param_fn = model_file

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    return net

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# regular, non-guided objective
def objective_L2(dst):
    dst.diff[:] = dst.data 

def objective_guide(dst):
    x = dst.data[0].copy()
    # guide_features is global here
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

class Dreamer(object):
    def __init__(self, net, source_path, iterations, guide_path):
        self.img = np.float32(PIL.Image.open(source_path))
        self.net = net
        self.iterations = iterations
        self.guide_func
        if guide_path:
            self.
        self.guide= guide
        #self.iterated_dream(source_path, iterations)

    # iterated dream and guided dream could prolly be combined
    # (guided is just a preprocess)
    def guided_dream(source_path, guide_path):
        img = np.float32(PIL.Image.open(source_path))
        guide = np.float32(PIL.Image.open(guide_path))
        #showarray(guide)
        end = 'inception_3b/output'
        h, w = guide.shape[:2]
        src, dst = net.blobs['data'], net.blobs[end]
        src.reshape(1,3,h,w)
        src.data[0] = preprocess(net, guide)
        net.forward(end=end)
        # global required for overwriting the guide features
        global guide_features
        # what we are doing here is setting the guide once and keeping it;
        #    just x in objective_guide is modified
        # the objective guide is first passed into deepdream and then into make_step!
        # end, on the other hand, is processed in forward and backward
        guide_features = dst.data[0].copy()
        result1 = deepdream(net, img, end=end, objective=objective_guide)

        PIL.Image.fromarray(np.uint8(result1)).save(get_output_path(source_path))

    def iterated_dream(self):
        #img = np.float32(PIL.Image.open(source_path))
        self.net.blobs.keys()

        frame = self.img

        h, w = frame.shape[:2]
        s = 0.05 # scale coefficient
        for i in xrange(self.iterations):
            frame = self.deepdream(frame)
            PIL.Image.fromarray(np.uint8(frame)).save(output_path())
            frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)

    def make_step(self, step_size=1.5, end='inception_4c/output', 
                  jitter=32, clip=True, objective=objective_L2):
        """Basic gradient ascent step."""

        src = self.net.blobs['data'] # input image is stored in Net's 'data' blob
        dst = self.net.blobs[end]

        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
                
        self.net.forward(end=end)
        objective(dst)  # specify the optimization objective
        self.net.backward(start=end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size/np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
                
        if clip:
            bias = self.net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255-bias)  

    def deepdream(self, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
                  end='inception_4c/output', clip=True, **step_params):
        # prepare base images for all octaves
        octaves = [preprocess(self.net, base_img)]
        for i in xrange(octave_n-1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
        
        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

            src.reshape(1,3,h,w) # resize the network's input image size
            src.data[0] = octave_base+detail
            for i in xrange(iter_n):
                self.make_step(end=end, clip=clip, **step_params)
                
                # visualization
                vis = deprocess(self.net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                #showarray(vis)
                # is octave, i the depth?
                print octave, i, end, vis.shape
                clear_output(wait=True)
                
            # extract details produced on the current octave
            detail = src.data[0]-octave_base
        # returning the resulting image
        return deprocess(self.net, src.data[0])

'''
def make_step(net, step_size=1.5, end='inception_4c/output', 
              jitter=32, clip=True, objective=objective_L2):
    """Basic gradient ascent step."""

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)  

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            #showarray(vis)
            # is octave, i the depth?
            print octave, i, end, vis.shape
            clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])
'''

def output_path():
    # faster with sort

    index=0
    output_file = "dreams/%06d.jpg"%index

    while os.path.exists(output_file):
        index += 1
        output_file = "dreams/%06d.jpg"%index

    return output_file


# enable --help functionality

# how does the "impressionist" style work?

# shallow layers create textured images:
# deepdream(net, img, end='inception_3b/5x5_reduce')
# why is this shallow?

# 3b is more shallow than 4c...
'''
def objective_guide(dst):
    x = dst.data[0].copy()
    # guide_features is global here
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

# iterated dream and guided dream could prolly be combined
# (guided is just a preprocess)
def guided_dream(source_path, guide_path):
    img = np.float32(PIL.Image.open(source_path))
    guide = np.float32(PIL.Image.open(guide_path))
    #showarray(guide)
    end = 'inception_3b/output'
    h, w = guide.shape[:2]
    src, dst = net.blobs['data'], net.blobs[end]
    src.reshape(1,3,h,w)
    src.data[0] = preprocess(net, guide)
    net.forward(end=end)
    # global required for overwriting the guide features
    global guide_features
    # what we are doing here is setting the guide once and keeping it;
    #    just x in objective_guide is modified
    # the objective guide is first passed into deepdream and then into make_step!
    # end, on the other hand, is processed in forward and backward
    guide_features = dst.data[0].copy()
    result1 = deepdream(net, img, end=end, objective=objective_guide)

    PIL.Image.fromarray(np.uint8(result1)).save(get_output_path(source_path))

def iterated_dream(source_path, iterations):
    img = np.float32(PIL.Image.open(source_path))
    net.blobs.keys()

    frame = img

    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient
    for i in xrange(int(iterations)):
        frame = deepdream(net, frame)
        PIL.Image.fromarray(np.uint8(frame)).save(output_path())
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
'''

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', nargs='?', const='sky_1024.jpg', default='sky_1024.jpg')
    parser.add_argument('-g', '--guide', nargs='?', default=None)
    parser.add_argument('-i', '--iterations', nargs='?', type=int, const=1, default=1)
    parser.add_argument('-m', '--model', nargs='?', metavar='int', type=int,
                                    choices=xrange(1, 6), help='model 1..5',
                                    const=1, default=1)
    # add -d = depth

    models_base = '../caffe/models'
    #model_default = '../caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    models = ('bvlc_googlenet/bvlc_googlenet.caffemodel',
                    'bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    'bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel',
                    'finetune_flickr_style/finetune_flickr_style.caffemodel',
                    'bvlc_alexnet/bvlc_alexnet.caffemodel')
    # add depth



    args = parser.parse_args(sys.argv[1:])
    net = create_net(os.path.join(models_base, models[args.model-1]))

    dreamer = Dreamer(net, args.source, args.iterations, args.guide)
    dreamer.iterated_dream()
    '''
    if args.guide:
        guided_dream(args.source, args.guide)
    else:
        iterated_dream(args.source, args.iterations)
    '''


    # sample input:
    # normal dream
    # python deepdreaming.py start=raspberry_pi_1024.jpg guide=None iterations=None
    # iterative dream
    # python deepdreaming.py start=raspberry_pi_1024.jpg guide=None iterations=2
    # guided dream
    # python deepdreaming.py start=raspberry_pi_1024.jpg guide=mops_1024.jpg iterations=None

    # can we fuse iteratvie and normal dream
    # plus, iterative guided dream???
    # getopt, optparse, argparse

    # ../caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel
    # ../caffe/models/googlelet_places/googlelet_places.caffemodel
    # ../caffe/models/nin_imagenet/nin_imagenet.caffemodel

    # ARTICLE:
    #   link to NN downloads, explain where to store, pass a model abbreviation
    #   OR pass full model path (check for robustness , what happens w/o a subfolder)
    # this may be the way to add models(see create_net) (and shortcut them in argv):
    # ./scripts/download_model_binary.py models/bvlc_googlenet

    # Change github repo readme

    # install additional nets:
    # cd ~/deepdream/caffe
    # ./scripts/download_model_binary.py models/bvlc_reference_caffenet
    # ./scripts/download_model_binary.py models/bvlc_reference_rcnn_ilsvrc13
    # ./scripts/download_model_binary.py models/finetune_flickr_style
    # ./scripts/download_model_binary.py models/bvlc_alexnet

    # the other models do not work right now...

    # make model an indexed parameter! [1..5] for the different types

    # add help:
    #   define guide for guided dreams
    #   define iterations as the number of iterations
    #   change the source with s
    #   do shallow dreams
    #   guide and iterations won't work now
    #   output is dream_XXX with XXX being the input filename


    # test arg parsing
    #print(parser.parse_args([]))
    #print(parser.parse_args(['-f']))
    #print(parser.parse_args('-f mops_1024.jpg -g sky_1024.jpg -i 100'.split()))
    # test arg parser with prior code