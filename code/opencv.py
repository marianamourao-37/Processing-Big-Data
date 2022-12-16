# Extract frames (refactor this function also)
# play

import numpy as np 
import time
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

import skeletons


def line(image, point1, point2,colour, width):
    #cv2.line(img, (x1, y1), (x2, y2), _DRAW_COLOUR, _DRAW_LINE_WIDTH)
    return cv2.line(image, point1, point2, colour, width)


def circle(image, point, radius, colour, thickness):
    #cv2.circle(img, (x2, y2), _DRAW_CIRCLE_RADIOUS, _DRAW_COLOUR, -1) 
    return cv2.circle(image, point, radius, colour, thickness) 


def extract_frames(from_path, frames, nrows_plot, ncols_plot, title_plot, save, to_path='./'):
  
    video = cv2.VideoCapture(from_path)
        
    fig, axs = plt.subplots(nrows_plot, ncols_plot, figsize=(15, 12))
    fig.suptitle(title_plot, fontsize=18, y=0.95)
    
    for num_frame, ax in zip(range(len(frames)), axs.ravel()):
        
        frame_num = frames[num_frame]
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num);
        ret, frame = video.read()
        
        if ret:            
            ax.set_title('frame '+ str(frame_num))
            ax.imshow(frame)
            ax.axis('off')
            
            if save:
                cv2.imwrite(to_path + 'frame_'+str(frame_num)+'.png', frame)

    video.release()
    cv2.destroyAllWindows()
    return True


def interactive_outlier_display(VIDEO_FILE, features, norm_orth_proj, outliers):

    def _plot(index, plotted, outlier, subspace_name):
        plt.subplot(2, 2, index)
        plt.plot(plotted)
        plt.plot(outlier, plotted[outlier], '*r', label = 'outlier')
        plt.legend(loc='lower right')
        plt.title('Norm of the images in the {0} subspace'.format(subspace_name))
        return True


    video = cv2.VideoCapture(VIDEO_FILE)

    df = np.linalg.norm(features, axis=0)

    for i in range(len(outliers)):
        clear_output(wait=True)

        fig = plt.figure(figsize=(15, 12))
        fig.suptitle('Outlier - frame {0}'.format(outliers[i]), fontsize=18, y=0.95)

        _plot(1, norm_orth_proj, outliers[i], 'null')
        #plt.subplot(2, 2, 1)
        #plt.plot(norm_orth_proj)
        #plt.plot(outliers[i], norm_orth_proj[outliers[i]], '*r', label = 'outlier')
        #plt.legend(loc='lower right')
        #plt.title('Norm of the images in the null subspace')

        _plot(2, df, outliers[i], 'basis')
        #plt.subplot(2, 2, 2)
        #plt.plot(df)
        #plt.plot(outliers[i], df[outliers[i]], '*r', label = 'outlier')
        #plt.legend(loc='lower right')
        #plt.title('Norm of the images in the basis subspace')

        plt.subplot(2, 1, 2)
        video.set(cv2.CAP_PROP_POS_FRAMES, outliers[i])
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)

        plt.show()
        time.sleep(0.3)

    video.release()
    cv2.destroyAllWindows()
    return True


def skeletons_frames_overlap(video_file, skel_data_tuple, frames, fps = 3, display = '', title_list = [], write_video = True, processed_video = 'processed_girosmllveryslow2.avi', write_video_path = 'processed_girosmllveryslow2_python.mp4'):

    def _show_frame(axis, image, title = ''):
        axis.imshow(image)
        if title != '':
            axis.title.set_text(title)
        return True


    original_video = cv2.VideoCapture(video_file)
    
    if display == 'processed':
        v_processed = cv2.VideoCapture(processed_video)
        
    elif write_video:
        original_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame = original_video.read()
        
        width = frame.shape[0]
        height = frame.shape[1]
        
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        write_vid = cv2.VideoWriter(write_video_path, fourcc, fps, (width, height))   
        # If write_video_path is constant then change it to setup file
    
    for frame_ix in frames:
        clear_output(wait=True)
        
        if display == 'reconstructed':
            fig, axs = plt.subplots(len(skel_data_tuple), 1, figsize=(15, 12))
            fig.suptitle('frame {0}'.format(frame_ix))
            
        for tuple_idx in range(len(skel_data_tuple)):
            
            skel = skel_data_tuple[tuple_idx]
            
            original_video.set(cv2.CAP_PROP_POS_FRAMES, frame_ix)
            _, frame = original_video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_with_poses = frame.copy()
        
            idx_frames = np.argwhere(skel[0, :] + 1 == frame_ix)
            pose = skel[:, idx_frames]
        
            if np.any(pose):
            
                skel_x = skeletons.skeletons_x(pose)
                skel_y = skeletons.skeletons_y(pose)
                skeletons.draw_poses(img_with_poses, skel_x, skel_y, len(idx_frames))
        
            if display == 'processed':
                
                fig, axs = plt.subplots(2, 1, figsize=(15, 12))
        
                fig.suptitle('frame {0}'.format(frame_ix))
            
                v_processed.set(cv2.CAP_PROP_POS_FRAMES, frame_ix)
                _, frame_processed = v_processed.read()
                frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                img_with_poses_processed = frame_processed.copy()
            
                _show_frame(axs[0], img_with_poses_processed, 'Processed')
                _show_frame(axs[1], img_with_poses, 'Skeletons Overlap in Python')
                #axs[0].imshow(img_with_poses_processed)
                #axs[0].title.set_text('Processed')
                #axs[1].imshow(img_with_poses)
                #axs[1].title.set_text('Skeletons Overlap in Python')
                
            if display == 'reconstructed':
                title = title_list[tuple_idx] if title_list else ''
                _show_frame(axs[tuple_idx], img_with_poses, title)
                
                #axs[tuple_idx].imshow(img_with_poses)
                
                #if title_list:
                #    axs[tuple_idx].title.set_text(title_list[tuple_idx])
                    
        if write_video:
            write_vid.write(img_with_poses)
            
        elif display != '':
            plt.show()
            time.sleep(1/fps)

    original_video.release()
    
    if write_video:
        write_vid.release()
    elif display == 'processed':
        v_processed.release()
        
    cv2.destroyAllWindows()
    return True


def display_segments(video_file, segment_frames):

    video = cv2.VideoCapture(video_file)

    for i in range(1,len(segment_frames),2):

        for idx_frame in range(segment_frames[i-1], segment_frames[i]+1):

            clear_output(wait=True)

            fig = plt.figure(figsize=(15, 12))
            fig.suptitle('Segment {0}'.format(int(np.ceil(i/2)), fontsize=18))

            video.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
            _, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
            plt.title('Frame {0}'.format(idx_frame))
            plt.show()

    video.release()
    cv2.destroyAllWindows()
    return True
