import numpy as np
import cv2
import time

# Algoorithm parameters
lk_params = dict(winSize = (15, 15),                                                                            # window size
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))                        # criteria for Lucas-Kannade method

# Feature parameters
feature_params = dict(maxCorners = 10,                                                                          # How many corners do we want to track
                     qualityLevel = 0.5,                                                                        # Quality of corners for tracking
                     minDistance = 30,                                                                          # Minimum distance between the corners we want to detect
                     blockSize = 7)


trajectory_len = 40             # how many points for each trajectory, lower the parameter if you don't want to track too long
detect_interval = 1             # how often to update the feature set
trajectories = []
frame_idx = 0

cap = cv2.VideoCapture(1)

while True:
    
    # start time to calculate FPS
    start = time.time()
    
    suc, frame = cap.read()                                 # read the frame from the webcam
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # convert to 8-bit input image with one channel
    img = frame.copy()
    
    # Calculate optical flow function for sparse feature set using Lucas-Kanade method
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)             # how the frame changes from previous frame (img0) to the current frame (img1)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)                                                   
        good = d < 1                                                                            # is the optical flow good enough
        
        new_trajectories = []
        
        # Get all the trajectories
        for trajectory, (x,y), good_flag in zip(trajectories, p1.reshape(-1,2), good):          # run through all the trajectories, take x,y points of the trajectories, check if it is good detection
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]                       # delete first point to only store the latest 20 (or whatever trajectory_len has been set up to)
            new_trajectories.append(trajectory)
            # Newest detected
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            
        trajectories = new_trajectories
        
        # Draw the trajectories
        cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
        cv2.putText(img, 'track count: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        
    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])


    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    
    # calculate the FPS for current frame detectionq
    fps = 1 / (end-start)
    
    # Show Results
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Optical Flow', img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        print(trajectories)
        break


cap.release()
cv2.destroyAllWindows()