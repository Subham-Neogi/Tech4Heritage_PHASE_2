''' 
This code is for inference.
To generate checkpoints the ../notebooks/Phase2_T4H_Pix2Pix_GAN.ipynb should be run first
''''



import argparse
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Generator:
    def __init__(self, dir):
        self.BUFFER_SIZE = 400
        self.BATCH_SIZE = 1
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.OUTPUT_CHANNELS = 3
        self.checkpoint_dir = dir
    
    def load(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        w = tf.shape(image)[1]

        w = w // 2
        real_image = image[:, :w, :]
        input_image = image[:, w:, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def resize(self, input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return input_image, real_image
    
    def normalize(self, input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1
        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = self.load(image_file)
        input_image, real_image = self.resize(input_image, real_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image, real_image = self.normalize(input_image, real_image)

        return input_image, real_image
    
    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result
    
    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result
    
    def generator(self, k):
        inputs = tf.keras.layers.Input(shape=[256,256,3])

        down_stack = [
            self.downsample(64,  k, apply_batchnorm=False), # (bs, 128, 128, 64)
            self.downsample(128, k), # (bs, 64, 64, 128)
            self.downsample(256, k), # (bs, 32, 32, 256)
            self.downsample(512, k), # (bs, 16, 16, 512)
            self.downsample(512, k), # (bs, 8, 8, 512)
            self.downsample(512, k), # (bs, 4, 4, 512)
            self.downsample(512, k), # (bs, 2, 2, 512)
            self.downsample(512, k), # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, k, apply_dropout=True), # (bs, 2, 2, 1024)
            self.upsample(512, k, apply_dropout=True), # (bs, 4, 4, 1024)
            self.upsample(512, k, apply_dropout=True), # (bs, 8, 8, 1024)
            self.upsample(512, k), # (bs, 16, 16, 1024)
            self.upsample(256, k), # (bs, 32, 32, 512)
            self.upsample(128, k), # (bs, 64, 64, 256)
            self.upsample(64,  k), # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh') # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def discriminator(self, k):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

        down1 = self.downsample(64,  k, False)(x) # (bs, 128, 128, 64)
        down2 = self.downsample(128, k)(down1) # (bs, 64, 64, 128)
        down3 = self.downsample(256, k)(down2) # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, k, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, k, strides=1,
                                        kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def process_image(self, image_inp, image_pred):
        bw_inp_img = cv2.cvtColor(image_inp, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(bw_inp_img,230,255,cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        
        img1 = cv2.bitwise_and(image_inp,image_inp,mask = inv_mask)
        img2 = cv2.bitwise_and(image_pred,image_pred,mask = mask)

        final_image = cv2.add(img1, img2)
        return final_image

    def generate_image(self, input_image_path, output_path):
        img = cv2.imread(input_image_path)
        h,w,c = img.shape

        vis = np.concatenate((img, img), axis=1)
        cv2.imwrite('./intermediate.jpg', vis)

        with tf.device('/cpu:0'):

            test_dataset = tf.data.Dataset.list_files('./intermediate.jpg')
            test_dataset = test_dataset.map(self.load_image_test)
            test_dataset = test_dataset.batch(self.BATCH_SIZE)

            generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

            model = self.generator(7)
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=model,
                                 discriminator=self.discriminator(7))
            manager = tf.train.CheckpointManager(checkpoint, self.checkpoint_dir, max_to_keep=3)
            checkpoint.restore(manager.latest_checkpoint)
            if  manager.latest_checkpoint:
                print("Restored first ckpt from {}".format(manager.latest_checkpoint))
            else:
                print("Failed to restore.")
                return 

            for input_image, _ in test_dataset.take(1):
                
                prediction = model(input_image, training=True)
                out =  cv2.cvtColor((prediction[0].numpy()*0.5+0.5)*255, cv2.COLOR_RGB2BGR)
                out = cv2.resize(out, (w, h), interpolation = cv2.INTER_CUBIC)
                out = out.astype(np.uint8)
                out = self.process_image(img,out)
                cv2.imwrite(output_path, out)
                


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate image with  inpainting of distortions.')
    parser.add_argument('-i', '--input', required=True, help='Input image file path')
    args = vars(parser.parse_args())
    outPath = './generated.jpg'
    dir = '../training_checkpoints'
    generator = Generator(dir)
    generator.generate_image(args['input'], outPath)