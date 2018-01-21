"""PixelCNN on Flickr8k.
"""
"""
PixelCNN Training code and utilities are licensed under APL2.0 from

Parag Mital
---------------------
https://github.com/pkmital/pycadl/blob/master/cadl/pixelcnn.py

Copyright 2018 Holly Grimm.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cadl import pixelcnn
from cadl import dataset_utils as dsu


def train(ckpt_path='pixelcnn',
                        n_epochs=1000,
                        save_step=100,
                        write_step=25,
                        B=4,
                        H=64,
                        W=64,
                        C=3):
    """Summary

    Parameters
    ----------
    ckpt_path : str, optional
        Description
    n_epochs : int, optional
        Description
    save_step : int, optional
        Description
    write_step : int, optional
        Description
    B : int, optional
        Description
    H : int, optional
        Description
    W : int, optional
        Description
    C : int, optional
        Description
    """
    ckpt_name = os.path.join(ckpt_path, 'pixelcnn.ckpt')

    with tf.Graph().as_default(), tf.Session() as sess:
        # Not actually conditioning on anything here just using the gated cnn model
        net = pixelcnn.build_conditional_pixel_cnn_model(B=B, H=H, W=W, C=C)

        # build the optimizer (this will take a while!)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001).minimize(net['cost'])

        # Load a list of files for flickr8k-64 dataset
        path = './flickr8k-64/'
        imagepath = os.path.join(path, 'Flicker8k_Dataset')
        tokenFilename =  os.path.join(path, 'Flickr8k.token.txt')

        with open(tokenFilename) as f:
            captiontxt = f.readlines()
        fs = []
        labels = []

        captiontxt_everyfifthline = captiontxt[::5]

        for line in captiontxt_everyfifthline:
            parts = line.split('\t')
            imgid = parts[0].split('.jpg')[0]
            label = parts[1][:-1].replace('\n','').strip()
            fs.append(os.path.join(imagepath, imgid + '.jpg'))
            labels.append(label)

        # Create a threaded image pipeline which will load/shuffle/crop/resize
        batch = dsu.create_input_pipeline(
            fs,
            batch_size=B,
            n_epochs=n_epochs,
            shape=[64, 64, 3],
            crop_shape=[H, W, C],
            crop_factor=1.0,
            n_threads=8)

        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(ckpt_path)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # This will handle our threaded image pipeline
        coord = tf.train.Coordinator()

        # Ensure no more changes to graph
        tf.get_default_graph().finalize()

        # Start up the queues for handling the image pipeline
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if os.path.exists(ckpt_name + '.index') or os.path.exists(ckpt_name):
            saver.restore(sess, ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        epoch_i = 0
        batch_i = 0
        try:
            while not coord.should_stop() and epoch_i < n_epochs:
                batch_i += 1
                batch_xs = sess.run(batch)
                train_cost = sess.run(
                    [net['cost'], optimizer], feed_dict={net['X']: batch_xs})[0]

                print(batch_i, train_cost)
                if batch_i % write_step == 0:
                    summary = sess.run(
                        net['summaries'], feed_dict={net['X']: batch_xs})
                    writer.add_summary(summary, batch_i)

                if batch_i % save_step == 0:
                    # Save the variables to disk.  Don't write the meta graph
                    # since we can use the code to create it, and it takes a long
                    # time to create the graph since it is so deep
                    saver.save(
                        sess,
                        ckpt_name,
                        global_step=batch_i,
                        write_meta_graph=True)
        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            # One of the threads has issued an exception.  So let's tell all the
            # threads to shutdown.
            coord.request_stop()

        # Wait until all threads have finished.
        coord.join(threads)

def generate(img_index=1):
    """Use PixelCNN to regenerate the bottom half of a selected image
    Parameters
    ----------
    img_index : int, optional
        Index into the list of image files
    """    
    # Parameters for generation
    ckpt_path = 'pixelcnn'
    B = None
    H = 64
    W = 64
    C = 3

    with tf.Graph().as_default(), tf.Session() as sess:
        # Not actually conditioning on anything here just using the gated cnn model
        net = pixelcnn.build_conditional_pixel_cnn_model(B=B, H=H, W=W, C=C)

        # Load a list of files for flickr8k
        path = './flickr8k-64/'
        imagepath = os.path.join(path, 'Flicker8k_Dataset')
        tokenFilename =  os.path.join(path, 'Flickr8k.token.txt')

        with open(tokenFilename) as f:
            captiontxt = f.readlines()
        fs = []
        labels = []

        captiontxt_everyfifthline = captiontxt[::5]

        for line in captiontxt_everyfifthline:
            parts = line.split('\t')
            imgid = parts[0].split('.jpg')[0]
            label = parts[1][:-1].replace('\n','').strip()
            fs.append(os.path.join(imagepath, imgid + '.jpg'))
            labels.append(label)

        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        # select one file to synthesize
        img = plt.imread(fs[img_index])
        from scipy.misc import imresize
        og_img = imresize(img, (H, W))
        img = og_img.copy()
        # Zero out bottom half of image and let's try to synthesize it
        img[H // 2:, :, :] = 0
        for h_i in range(H // 2, H):
            for w_i in range(W):

#         # Zero out right half of image and let's try to synthesize it
#         img[:, W // 2:, :] = 0
#         for h_i in range(H):
#             for w_i in range(W // 2, W):
                for c_i in range(C):
                    print(h_i, w_i, c_i, end='\r')
                    X = img.copy()
                    preds = sess.run(
                        net['sampled_preds'],
                        feed_dict={net['X']: X[np.newaxis]})
                    X = preds.reshape((1, H, W, C)).astype(np.uint8)
                    img[h_i, w_i, c_i] = X[0, h_i, w_i, c_i]
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(og_img)
        axs[1].imshow(img)
        plt.show()


if __name__ == '__main__':
    train()