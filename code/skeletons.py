# 
## skel_info
## This: # ALTERNATIVA AO BAR PLOT - RECORRER DIRETAMENTE A UM HISTOGRAMA: (put 
## it in a fuction)
# extract_skeletons
# get_joint
# draw_poses
## make use of some function in plotting.py
# call staking data functions from data_preprocessing.py to stake the skeletons
import numpy as np

import plotting


def skeletons_coords(skeletons):
    skel_x = skeletons_x(skeletons)
    skel_y = skeletons_y(skeletons)

    return np.vstack((skel_x, skel_y))


def extract_skeletons(all_skeletons, image):
    """ Get skeletons of a given image."""
    print('Selecting skeletons from frames ', all_skeletons[0])
    indexes = np.argwhere(all_skeletons[0] == image).reshape(-1)
    skeletons = np.array([all_skeletons[:, skeleton] for skeleton in indexes]).T
    return skeletons


def get_joint(skeleton, joint_nmb):
    """ Get joint of a specific skeleton."""
    print(skeleton)
    joint_range = range(3 * joint_nmb, 3 * (joint_nmb + 1))
    return np.array([skeleton[feature] for feature in joint_range])


def skeletons_x(skeletons):
    return skeletons[1::3, :]


def skeletons_y(skeletons):
    return skeletons[2::3, :]


def skel_info(data):
    last = int(data[0][-1])
    frames = [i for i in range(last + 1)]
    hist = [0] * (last + 1)
    for i in data[0]:
        hist[int(i)] += 1
    plotting.bar_plot(frames, hist, 'Skels per frame', 'Frame', '# of Skels')
    return True


def draw_pose(img, skeletons_x, skeletons_y, display_missing_data = False):
    return plotting.draw_pose(img, skeletons_x, skeletons_y, display_missing_data)

    ## FIXME set these as constant in setup file
    #_PAIRS = [
    #(0, 1), (0, 14), (0, 15), (1, 2), (1, 5), (1, 8), (1, 11), #torso 
    #    (2, 3), (3, 4), # right arm 
    #    (5, 6), (6, 7), #left arm
    #    (8, 9), (9, 10), #right leg
    #    (11, 12), (12, 13), #lef leg
    #    (14, 16), (15, 17),  # face
    #]
    #
    #_DRAW_COLOUR = (255, 0, 0)
    #_DRAW_CIRCLE_RADIOUS = 3
    #_DRAW_HAND_RADIOUS = 5
    #
    #_DRAW_LINE_WIDTH = 2
    #_DRAW_COLOUR_HANDS = (0,0,255)
#
#
    #
    #img_height = img.shape[0]
    #img_width = img.shape[1]
    #
#
    #valid = (skeletons_x + skeletons_y).astype(bool) # PODE SE TIRAR SE QUISEREM MESMO VER A MISSING DATA 
#
    #for kp1, kp2 in _PAIRS:
    #    
    #    # PODE SE TIRAR ESTE IF SE QUISEREM VER A MISSING DATA - I think it's fixed with the OR, couldn't test tho
    #    if display_missing_data or valid[kp1] & valid[kp2]: 
    #        x1 = int(skeletons_x[kp1, :] * img_width)
    #        y1 = int(skeletons_y[kp1, :] * img_height)
    #        x2 = int(skeletons_x[kp2, :] * img_width)
    #        y2 = int(skeletons_y[kp2, :] * img_height)
    #    
    #        opencv.line(img, (x1, y1), (x2, y2), _DRAW_COLOUR, _DRAW_LINE_WIDTH)
    #        #cv2.line(img, (x1, y1), (x2, y2), _DRAW_COLOUR, _DRAW_LINE_WIDTH)
    #    
    #        #if kp2 == 4 or kp2 == 7:
    #        #    cv2.circle(img, (x1, y1), _DRAW_CIRCLE_RADIOUS, _DRAW_HAND_RADIOUS, -1)
    #        #else:
#
    #        opencv.circle(img, (x1, y1), _DRAW_CIRCLE_RADIOUS, _DRAW_COLOUR, -1)
    #        opencv.circle(img, (x2, y2), _DRAW_CIRCLE_RADIOUS, _DRAW_COLOUR, -1)  
    #        #cv2.circle(img, (x1, y1), _DRAW_CIRCLE_RADIOUS, _DRAW_COLOUR, -1)
    #        #cv2.circle(img, (x2, y2), _DRAW_CIRCLE_RADIOUS, _DRAW_COLOUR, -1)  
#
    #return True
    #        

def draw_poses(img, skeletons_x, skeletons_y, num_poses):

    num_joints = skeletons_x.shape[0] # if it ends up being a constant then move it to setup
    for pose in range(num_poses):
        draw_pose(img, skeletons_x[:, pose].reshape(num_joints, 1), skeletons_y[:, pose].reshape(num_joints, 1))
    return True
    


if __name__ == '__main__':

    import data_preprocessing as dp
    import setup
    import opencv

    frame = [2938]

    incomplete_skeletons_dev = setup.INCOMPLETE_SKELS_FRAME_DEV
    #incomplete_skeletons_dev = np.array([[0,2,0,7,6,2], [4,5,0,6,6,1], [5,6,1,5,3,2], [5,1,4,2,6,2], [1,4,7,3,8,7]])
    
    skel_pos = skeletons_coords(incomplete_skeletons_dev)
    skels = extract_skeletons(incomplete_skeletons_dev, frame)
    draw_poses(skeletons_x(skels), skeletons_y(skels))

    # save frame to folder
    opencv.extract_frames(setup.VIDEO_FILE, frame, 4, 3, '{0} frames'.format(frame), True, setup.IMAGES_DIR)
