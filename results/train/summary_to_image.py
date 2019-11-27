import os
import argparse
import scipy.misc
import tensorflow as tf

def save_images_from_event(fn, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
        help="summary path")
    ap.add_argument("-t", "--tag", required=True,
        help="name of the tag")
    args = vars(ap.parse_args())
    
    # call function
    save_images_from_event(args['path'], args['tag'])


if __name__ == '__main__':
    main()