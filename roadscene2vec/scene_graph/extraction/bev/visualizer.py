from pathlib import Path

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from bev import BEV

'''
    BEV Visualizer: run this script to view the bird's eye projected version of your image dataset
    To be used after calibrating BEV projection in bev.py | BEV(path, mode='calibrate')
'''

def read_int(fname):
    with open(str(fname), 'r') as f:
        return int(f.read())

def skip_clip(clip):
    ignore_path = (clip/"ignore.txt").resolve()
    if ignore_path.exists():
        if read_int(ignore_path): return True
    return False

def bev_demo(bev, clip_path):
    M = bev.compute_homography_matrix()
    clips = [c for c in Path(clip_path).iterdir()]

    for clip in clips:
        if skip_clip(clip): continue
        vid = []
        counter = 1
        fig = plt.figure()
        frames = [f for f in (clip / "raw_images").iterdir()]
        print("Now showing: {}".format(str(clip)))
        
        for frame in frames:
            print(frame)
            img = cv2.imread(str(frame), cv2.IMREAD_UNCHANGED)
            assert (bev.params['height'], bev.params['width']) == img.shape[:-1]
            
            # images
            img = cv2.copyMakeBorder(img, bev.params['cropped_height'], 0, 0, 0, cv2.BORDER_CONSTANT)
            warped_img = cv2.warpPerspective(img, M, (bev.params['width'], bev.params['height']+bev.params['cropped_height'])) # Image warping
            merge_imgs = np.hstack((img, warped_img))
            
            # horizon line
            x = [0, 2*(bev.params['width']) - 1]
            y = [bev.params['horizon_height'], bev.params['horizon_height']]

            vid.append([plt.imshow(cv2.cvtColor(merge_imgs, cv2.COLOR_BGR2RGB)), plt.plot(x, y, color='red')[0]])

            if counter % 100 == 0:
                ani = animation.ArtistAnimation(fig, vid, interval=200, blit=True, repeat_delay=1000)
                plt.show()
                fig = plt.figure()
                vid = []

            counter +=1

if __name__ == "__main__":
    clip_path = "./honda"
    bev = BEV('bev.json', mode='deploy')
    bev_demo(bev, clip_path)