import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm
import itertools
from joint_bilateral_filter import Joint_bilateral_filter
from scipy.spatial import distance
from joblib import Parallel, delayed
import multiprocessing
from tools import *

class adv_rgb2gray:
    def __init__(self):
        self.sigma_s = [1., 2., 3.]
        self.sigma_r = [0.05, 0.1, 0.2]
        self.points = self._consturct_2D()
        self.neighbor_idx_tab = self._consturct_neighbor()
        self.result = []

    def _consturct_2D(self):
        elements = [i/10. for i in range(11)]
        all_combs = list(itertools.product(elements,repeat=3))
        points = []
        for ele in all_combs:
            if round(np.sum(ele),2) == 1.: points.append(ele)
        return points

    def _consturct_neighbor(self):
        neighbor_tmp = {}
        for idx in range(len(self.points)):
            ref = self.points[idx]
            tmp = []
            for idx_target in range(len(self.points)):
                target = self.points[idx_target]
                if round(distance.euclidean(ref,target),5) == 0.14142:
                    tmp.append(idx_target)
            neighbor_tmp[idx] = tmp
        return neighbor_tmp

    def conv_rgb2gray(self, img_rgb, w):
        '''
        arg : w = (wr, wg, wb)
        '''
        output = w[0] * img_rgb[:,:,0] + w[1] * img_rgb[:,:,1] \
                + w[2] * img_rgb[:,:,2]
        return output

    def run_sigle(self, scores, img_rgb, sigma_s, sigma_r):
        cost = []
        JBF = Joint_bilateral_filter(sigma_s, sigma_r, border_type='reflect')
        I_bf = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.float64)
        for idx, point in enumerate(tqdm(self.points,desc = "[s:{}, r:{}]".format(sigma_s,sigma_r))):
            guidance = self.conv_rgb2gray(img_rgb, point)
            I_jbf = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.float64)
            cost.append(np.sum(np.abs(I_bf-I_jbf)))
        # calculate the local min
        for idx in range(len(self.points)):
            neighbors_idx = self.neighbor_idx_tab[idx]
            if cost[idx] <= np.array(cost)[neighbors_idx].min():
                scores[idx] += 1
        return scores

    def run_base(self, path, output_path):
        print('Processing img \'{}\' ......'.format(path))
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scores = [0 for i in range(len(self.points))]
        for s in self.sigma_s:
            for r in self.sigma_r:
                scores = self.run_sigle(scores, img_rgb, s, r)
        # save conv img & result
        img_num = path.split('/')[-1][:2]
        img_type = path.split('/')[-1][2:]
        weight_score_dict = {}
        sort_idx = np.argsort(scores)
        for i in [1,2,3]:
            weight_score_dict[self.points[sort_idx[-1*i]]] = scores[sort_idx[-1*i]]
            img_gray = self.conv_rgb2gray(img_rgb, self.points[sort_idx[-1*i]])
            image_name = img_num + '_y{}'.format(i) + img_type
            self.save_img(img_gray, output_path, image_name)

        self.result.append(weight_score_dict)

        print('Processing img \'{}\' ......done'.format(path))
        return True

    def run(self, img_path_list, output_path):
        num_cores = multiprocessing.cpu_count()
        #self.run_base(img_path_list[2], output_path, parallel_num = 0)
        for idx, path in enumerate(img_path_list):
            self.run_base(path, output_path)

        self.show_result()
        return False

    def show_result(self):
        print("Result : ")
        print("=> Total image number : ", len(self.result))
        for i in range(len(self.result)):
            pair = self.result[i]
            keys = list(pair.keys())
            print("=> Image {}".format(i+1))
            print("===> [Fisrt]  w : {} | score : {}".format(keys[0],pair[keys[0]]))
            print("===> [Second] w : {} | score : {}".format(keys[1],pair[keys[1]]))
            print("===> [Third]  w : {} | score : {}".format(keys[2],pair[keys[2]]))
        return False

    def verify(self, img_path, sigma_s_l, sigma_r_l, w):
        """
        args
            sigma_s_l: (list)
            sigma_r_l: (list)
        """
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        score = 0.
        sigma = pc_combine(sigma_s_l, sigma_r_l)

        w_neighbor = np.array(self.points)[self.neighbor_idx_tab[self.points.index(w)]]
        for i, (sigma_s, sigma_r) in enumerate(tqdm(sigma,desc = "[s:{}, r:{}]".format(sigma[0],sigma[1]))):
            JBF = Joint_bilateral_filter(sigma_s, sigma_r, border_type='reflect')
            I_bf = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.float64)
            # reference
            guidance = self.conv_rgb2gray(img_rgb, w)
            I_jbf = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.float64)
            cost_ref = np.sum(np.abs(I_bf-I_jbf))
            cost = []
            # target
            for idx, point in enumerate(w_neighbor):
                guidance = self.conv_rgb2gray(img_rgb, point)
                I_jbf = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.float64)
                cost.append(np.sum(np.abs(I_bf-I_jbf)))
            # calculate the local min
            if cost_ref <= np.array(cost).min():
                score += 1
        return score

    def save_img(self, img, output_path, img_name):
        img_path = os.path.join(output_path,img_name)
        cv2.imwrite(img_path,img)
        print("Save image in : {}".format(img_path))
        return True


def main():

    parser = argparse.ArgumentParser(description='Advanced rgb2gray')
    parser.add_argument('--input_path', default='./testdata', help='path of input image')
    parser.add_argument('--img_name', default=['0a.png','0b.png','0c.png'], help='(list)name of input image')
    parser.add_argument('--output_path', default='./save_img/', help='path of output image')
    args = parser.parse_args()


    print('<'+"="*20+'>')
    print("Input image from : {}".format(args.input_path))
    print("Input image is : {}".format(args.img_name))
    # Check file existance
    img_path_list = []
    for i in args.img_name:
        file = os.path.join(args.input_path,i)
        img_path_list.append(file)
        assert os.path.isfile(file) , 'The file \'{}\' is no exist in the directory.'.format(i)

    converter = adv_rgb2gray()
    converter.run(img_path_list, args.output_path)

    # output = converter.rgb2gray(args.input_path)
    print('<'+"="*20+'>')


def test():
    parser = argparse.ArgumentParser(description='Advanced rgb2gray')
    parser.add_argument('--input_path', default='./testdata', help='path of input image')
    parser.add_argument('--img_name', default=['0a.png','0b.png','0c.png'], help='(list)name of input image')
    parser.add_argument('--output_path', default='./save_img/', help='path of output image')
    args = parser.parse_args()


    print('<'+"="*20+'>')
    print("Input image from : {}".format(args.input_path))
    print("Input image is : {}".format(args.img_name))
    # Check file existance
    img_path_list = []
    for i in args.img_name:
        file = os.path.join(args.input_path,i)
        img_path_list.append(file)
        assert os.path.isfile(file) , 'The file \'{}\' is no exist in the directory.'.format(i)

    sigma_s_l = [1., 2., 3.]
    sigma_r_l = [0.05, 0.1, 0.2]
    converter = adv_rgb2gray()
    # Image 01
    results_w = [[(1.0,0.,0.),(0.,0.,1.0),(0.9,0.,0.1)],
                 [(1.0,0.,0.),(0.8,0.2,0.),(0.7,0.2,0.1)],
                 [(1.,0.,0.),(0.3,0.4,0.3),(0.3,0.5,0.2)]]
    results_score = [[9,9,0],
                     [3,3,2],
                     [3,3,3]]


    for img_id ,(img_path,w_list, score_list) in enumerate(zip(img_path_list,results_w,results_score)):
        for j in range(3):
            myscore = converter.verify(img_path, sigma_s_l, sigma_r_l, w_list[j])
            print("Image {}-{} get score {} ......{}".format(img_id+1,j+1,
                    myscore, (myscore == score_list[j])))

    print('<'+"="*20+'>')

if __name__ == '__main__':
    test()
