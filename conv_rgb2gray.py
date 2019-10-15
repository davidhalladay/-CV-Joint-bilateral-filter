import numpy as np
import cv2
import argparse
import os

class conv_rgb2gray:
    def __init__(self, p_r, p_g, p_b):

        self.p_r = p_r
        self.p_g = p_g
        self.p_b = p_b

    def rgb2gray(self, input_path):
        assert type(input_path) == type(str())

        img = cv2.imread(input_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # output = np.zeros((img_rgb.shape[0],img_rgb.shape[1]))

        output = self.p_r * img_rgb[:,:,0] + self.p_g * img_rgb[:,:,1] \
                + self.p_b * img_rgb[:,:,2]

        return output

    def save_img(self, img, output_path, img_name):
        img_path = os.path.join(output_path,img_name)
        cv2.imwrite(img_path,img)
        print("Save image in : {}".format(img_path))

        return True


def main():

    parser = argparse.ArgumentParser(description='Conventional rgb2gray')
    parser.add_argument('--input_path', default='./testdata/0c.png', help='path of input image')
    parser.add_argument('--output_path', default='./save_img/', help='path of output image')
    args = parser.parse_args()


    print('<'+"="*20+'>')

    print("Input image from : {}".format(args.input_path))

    converter = conv_rgb2gray(p_r = 0.299, p_g = 0.587, p_b = 0.114)
    output = converter.rgb2gray(args.input_path)

    converter.save_img(output,args.output_path,'1_0c.png')
    print('<'+"="*20+'>')

if __name__ == '__main__':
    main()
