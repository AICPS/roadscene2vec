import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


class BEV():
    def __init__(self, fname, mode='calibrate'):
        if mode == 'deploy':
            is_json = Path(fname).is_file() and '.json' in fname
            assert is_json, 'ERROR! file {} does not exist or is not a json file...'.format(fname)
            self.params = self.load_params(fname) # bev params
            self.compute_homography_matrix()

        if mode == 'calibrate':
            is_img = Path(fname).is_file() and ('.jpg' in fname or '.png' in fname)
            assert is_img, 'ERROR! file {} does not exist or is not an image...'.format(fname)
            self.params = {'proj_ratio': 3} # bev params
            self.fname = fname              # file
            self.lane_length = 10
            self.lane_width  = 12
            
            # clickable point(s)
            self.point = None               
            self.lane_points = [{}, {}, {}]

    def read_img(self):
        self.img = cv2.imread(self.fname, cv2.IMREAD_UNCHANGED)
        height, width = self.img.shape[:-1]
        self.params['height'] = height
        self.params['width'] = width

    def get_point(self):
        return self.point

    def set_point(self, event):
        self.point = int(event.ydata)

    def index_of_point(self, points, point):
        try:
            return points.index(point)
        except ValueError:    
            return None
        
    def is_none(self, item):
        return item == None

    def abs_diff(self, key, points):
        return abs(points[0][key] - points[1][key])

    def set_lane_points(self, event):
        i = self.index_of_point(self.lane_points, {})
        if not self.is_none(i):
            self.lane_points[i] = {'xdata': event.xdata, 'ydata': event.ydata}
            if self.is_none(self.index_of_point(self.lane_points, {})):
                self.params['yscale'] = float('%.3f' % (self.lane_length / self.abs_diff('ydata', self.lane_points[:-1])))
                self.params['xscale'] = float('%.3f' % (self.lane_width  / self.abs_diff('xdata', self.lane_points[1:])))

                # save params after clicking on lane lines
                self.save_params()

    def onclick(self, event):
        if not self.is_none(event.xdata) and not self.is_none(event.ydata):
            if self.is_none(self.point):
                self.set_point(event)
            else:
                self.set_lane_points(event)
            self.display_bev()

    def keydown(self, event):
        if event.key == 'r':
            self.reset_display()
        if not self.is_none(self.point):
            if event.key == 'e':
                self.sharper_proj()
            elif event.key == 'w':
                self.softer_proj()
            self.display_bev()

    def sharper_proj(self):
        self.params['proj_ratio'] += 2
        
    def softer_proj(self):
        if self.params['proj_ratio'] > 3:
            self.params['proj_ratio'] -= 2
        else:
            print('Cannot widen any further!')

    def reset_display(self):
        self.point = None
        self.lane_points = [{} for _ in self.lane_points]
        if 'xscale' in self.params:
            del self.params['xscale'], self.params['yscale']
        
        plt.clf()
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), animated=True)
        self.show_instructions()
        plt.draw()

    def save_params(self, fname='bev.json'):
        metadata = {}
        for k in self.params.keys():
            metadata[k] = self.params[k]

        with open(fname, 'w') as f:
            json.dump(metadata, f, indent=4)
            print('- saved params to {}'.format(fname))

    def load_params(self, fname):
        with open(fname, 'r') as f:
            return json.load(f)

    def show_params(self):
        parameters = ''
        for k in self.params.keys():
            parameters += '{}={}, '.format(k, self.params[k])
        plt.figtext(0.5, 0.1, parameters[: -2], ha='center', weight='bold', bbox=dict(boxstyle='square', facecolor='gainsboro', alpha=0.5), wrap=True)

    def apply_depth_estimation(self, x, y):
        return (x * self.params['xscale'], y * self.params['yscale'])
    
    def get_projected_point(self, x, y):
        # Replace line 140 image_scenegraph.py with call to this method
        point = np.array([[[x, y + self.params['cropped_height']]]], dtype='float32')
        return cv2.perspectiveTransform(point, self.M).squeeze()
        
    def compute_homography_matrix(self):
        # Set ROI coords
        padded_height      = self.params['height'] + self.params['cropped_height']
        self.bottom_left   = [0, padded_height]
        self.bottom_right  = [self.params['width'], padded_height]
        self.top_left      = [0, self.params['height']]
        self.top_right     = [self.params['width'], self.params['height']]
        src = np.float32([self.bottom_left, self.bottom_right, self.top_left, self.top_right])
        
        # Projection domain
        left_ratio  = math.floor(self.params['proj_ratio']/2) / self.params['proj_ratio']
        right_ratio = math.ceil(self.params['proj_ratio']/2)  / self.params['proj_ratio']
        bottom_left_ratio = int(self.params['width']*left_ratio)
        bottom_right_ratio = int(self.params['width']*right_ratio)
        
        # Create v-shape projection
        dst = np.float32([[bottom_left_ratio, padded_height], [bottom_right_ratio, padded_height], self.top_left, self.top_right]) 
        
        # transformation matrices
        self.M = cv2.getPerspectiveTransform(src, dst) 
        self.Minv = cv2.getPerspectiveTransform(dst, src) 

    def demo_points(self):
        # Show sample points offset from horizon line
        points = lambda x: x if type(x) == list else x.squeeze()
        plot = lambda x, c, m : plt.plot(*points(x), color=c, marker=m, ms=7)
        shift = lambda x: [x[0] + self.params['width'] - 1, x[1]]

        bottom  = np.array([[[self.params['width']/2, self.params['horizon_height'] + 20]]], dtype='float32')
        top     = np.array([[[self.params['width']/2, self.params['horizon_height'] - 20]]], dtype='float32')
        bottom_ = shift(cv2.perspectiveTransform(bottom, self.M).squeeze())
        top_    = shift(cv2.perspectiveTransform(top, self.M).squeeze())

        plot(bottom, c='orange', m='*')
        plot(top, c='green', m='*')
        plot(bottom_, c='orange', m='*')
        plot(top_, c='green', m='*')

    def warpPerspective(self, img):
        return cv2.warpPerspective(img, self.M, (self.params['width'], self.params['height'] + self.params['cropped_height']))
    
    def offset_image_height(self, img):
        return cv2.copyMakeBorder(img, self.params['cropped_height'], 0, 0, 0, cv2.BORDER_CONSTANT)

    def display_bev(self):
        cropped_top = self.get_point()
        # remove delete_me stuff
        self.params['cropped_height'] = self.params['height'] - cropped_top
        self.params['horizon_height'] = self.params['cropped_height'] + cropped_top
        self.compute_homography_matrix()

        # Apply np slicing for ROI crop (show on image)
        img = self.offset_image_height(self.img.copy())
        warped_img = self.warpPerspective(img) # Image warping
        merge_imgs = np.hstack((img, warped_img))
        plt.clf()
        plt.imshow(cv2.cvtColor(merge_imgs, cv2.COLOR_BGR2RGB), animated=True) # Show results

        # Show chosen horizon line
        x = [0, 2*(self.params['width']) - 1]
        y = [self.params['horizon_height'], self.params['horizon_height']]
        plt.plot(x, y, color='red')

        # Show user clicked points
        for point in self.lane_points:
            if 'xdata' in point:
                # Projected image points
                plt.plot(point['xdata'], point['ydata'], color='red', marker='o', ms=5)

                # Original image points
                x_inv, y_inv = cv2.perspectiveTransform(np.array([[[point['xdata'] - self.params['width'], point['ydata']]]]), self.Minv).squeeze()
                plt.plot(x_inv, y_inv, color='red', marker='o', ms=5)
            
        # self.demo_points()
    
        self.show_instructions(bev=True)
        self.show_params()
        plt.draw()

    def show_instructions(self, bev=False):
        if not bev:
            instructions = '''Find and click on the image's horizon line'''
            plt.annotate(instructions, (self.params['width']/2, -self.params['height']/30), annotation_clip=False, ha='center', wrap=True)
        else: 
            instructions = '''Original image'''
            plt.annotate(instructions, (self.params['width']/2, -self.params['height']/30), annotation_clip=False, ha='center', wrap=True)

            instructions = '''Projected image'''
            plt.annotate(instructions, (self.params['width']*3/2, -self.params['height']/30), annotation_clip=False, ha='center', wrap=True)

            instructions = '''Keypress [e] to elongate perspective, [w] to widen persepctive, [r] to reset image'''
            plt.figtext(0.5, 0.01, instructions, ha='center', weight='bold', bbox=dict(boxstyle='square', facecolor='gainsboro', alpha=0.5), wrap=True)

    def calibrate(self):
        self.read_img()
        fig = plt.figure()
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), animated=True)
        set_point = fig.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event))
        update_display = fig.canvas.mpl_connect('key_press_event', lambda event: self.keydown(event))
        self.show_instructions()
        plt.show()

if __name__ == '__main__':
    ap = ArgumentParser(description='The parameters for training.')
    ap.add_argument('--cal_im_path', type=str, default='/media/NAS-temp/louisccc/av/synthesis_data/1043_carla/22_lanechange/raw_images/00097095.jpg', help="The path defining location of image used to calibrate BEV.")
    path = ap.parse_args().cal_im_path
    bev = BEV(path, mode='calibrate')
    bev.calibrate()