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

#OUTPUT_DIR = "dreams/"

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

# lalalalfasfddf

# first step: get it to produce an image

'''
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    # why are these images not saved?
    # because f is of type StringIO and not a filename
    #display(Image(data=f.getvalue()))
'''

# make this a function: set_model
# make ability to switch models
def create_net(model_path):
    #model_path = '../caffe/models/bvlc_googlenet/' # substitute your path here
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + 'bvlc_googlenet.caffemodel'

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

def objective_L2(dst):
    dst.diff[:] = dst.data 

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


def objective_guide(dst):
    x = dst.data[0].copy()
    # guide_features is global here
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

def output_path():
    """ Create an output filename with the pattern 
         'dream_' + input_file + '_' + index + '.jpg'
         with index being the lowest number which does not overwrite an
         existing dream. """

    # prettify this nasty shite

    #output_file = OUTPUT_DIR + 'dream_' + input_file.split('.')[0] + '_1.jpg'
    index=0
    output_file = "dreams/%06d.jpg"%index
    # make this more simple, it should just be the image with the highest index.

    while os.path.exists(output_file):
        index += 1
        output_file = "dreams/%06d.jpg"%index

    return output_file


# enable --help functionality

# showarray prolly can be deleted

# how does the "impressionist" style work?

# shallow layers create textured images:
# deepdream(net, img, end='inception_3b/5x5_reduce')
# why is this shallow?

# 3b is more shallow than 4c...



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
    guide_features = dst.data[0].copy()
    result1 = deepdream(net, img, end=end, objective=objective_guide)

    PIL.Image.fromarray(np.uint8(result1)).save(get_output_path(source_path))

def iterated_dream(source_path, iterations):
    img = np.float32(PIL.Image.open(source_path))
    net.blobs.keys()

    frame = img
    #frame_i = 0

    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient
    for i in xrange(int(iterations)):
        frame = deepdream(net, frame)
        PIL.Image.fromarray(np.uint8(frame)).save(output_path())
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        #frame_i += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', nargs='?', const='sky_1024.jpg', default='sky_1024.jpg')
    parser.add_argument('-g', '--guide', nargs='?', default=None)
    parser.add_argument('-i', '--iterations', nargs='?', const=1, default=1)
    parser.add_argument('-m', '--model', nargs='?', const='../caffe/models/bvlc_googlenet/', default='../caffe/models/bvlc_googlenet/')
    # add depth

    # add help:
    #   define guide for guided dreams
    #   define iterations as the number of iterations
    #   change the source with s
    #   do shallow dreams
    #   guide and iterations won't work now
    #   output is dream_XXX with XXX being the input filename

    # output filename is dream_ + input_file + highest index which does not overwrite
    # a dream + .jpg


    # test arg parsing
    #print(parser.parse_args([]))
    #print(parser.parse_args(['-f']))
    #print(parser.parse_args('-f mops_1024.jpg -g sky_1024.jpg -i 100'.split()))
    # test arg parser with prior code

    # add hint about obtaining layers to article

    # we did not apply the different models!

    args = parser.parse_args(sys.argv[1:])

    net = create_net(args.model)
    

    if args.guide:
        guided_dream(args.source, args.guide)
    else:
        iterated_dream(args.source, args.iterations)


    # input arguments:
    #   optional input file -f --file default sky_1024.jpg
    #   optional -g --guide default None
    #   optional -i --iterations default 1
    #   optional -d --depth default=? some kind of measure of shallowness...

    # before implementing this, test if iterations=1 is the same as simple dream
    # yah, iterations = 1 is good!

    #print(sys.argv)
    #print(**sys.argv[1:])
    #start_dream(*sys.argv[1:])

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