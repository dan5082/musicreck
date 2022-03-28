import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy.linalg as la
from IPython.display import display, Image
import ipywidgets as widgets
import threading
import tesserocr
from PIL import Image
cv=cv2

cap = cv2.VideoCapture(0)
cap.set(3,640) # adjust width
cap.set(4,480) # adjust height

frame = 0
frame_skip = 1

left = ''
left_pos = (0,0)
top = ''
top_pos = (0,0)
bottom = ''
bottom_pos = (0,0)
right = ''
right_pos = (0,0)
winning_choice = ''


overall_winning = set()


def convert_image_region_to_text(clip):
    thresh2 = clip
    ret2,thresh2 = cv2.threshold(thresh2,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh2 = cv.Canny(thresh2, 100, 200)
    text = tesserocr.image_to_text(Image.fromarray(thresh2)).strip()
    text = text.replace('\n', '')
    return text, thresh2


while True:
    success, img = cap.read()

    frame += 1
    if frame % frame_skip != 0:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    mask = cv2.inRange(hsv, (0, 220, 0), (255, 255, 255))
    mask = cv2.dilate(mask, rect_kernel, iterations=1)
    mask = cv2.erode(mask, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w < 150 and h < 150:
            continue

        # if w > 640/2:
        #     continue

        filtered_contours.append(cnt)
    contours = filtered_contours    
    
    if len(contours) == 4:
        titles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            side_clip = 20
            x = x + side_clip
            w = w - side_clip * 2

            text, new_mask = convert_image_region_to_text(gray[y:y+h, x:x+w])
            mask[y:y+h, x:x+w] = new_mask
            titles.append((x+w/2, y+h/w, text))
            rect = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 0), 2)
        
        n_left = min(titles, key=lambda t: t[0])
        n_right = max(titles, key=lambda t: t[0])
        n_top = min(titles, key=lambda t: t[1])
        n_bottom = max(titles, key=lambda t: t[1])

        if len(n_left) > 0:
            left = n_left[2]
            left_pos = n_left[:2]
        if len(n_right) > 0:
            right = n_right[2]
            right_pos = n_right[:2]
        if len(n_top) > 0:
            top = n_top[2]
            top_pos = n_top[:2]
        if len(n_bottom) > 0:
            bottom = n_bottom[2]
            bottom_pos = n_bottom[:2]

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    green_mask = cv2.inRange(hsv, (30, 100, 100), (70, 200, 200))
    green_mask_orig = green_mask
    green_mask = cv2.dilate(green_mask, rect_kernel, iterations=1)
    green_mask = cv2.erode(green_mask, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        pos = np.array([x+w/2, y+h/2])
        other_pos = np.array([
            left_pos, top_pos, bottom_pos, right_pos
        ])
        nearest = np.argmin(la.norm(other_pos-pos, 2, axis=1))
        
        if nearest == 0:
            winning_choice = left
        elif nearest == 1:
            winning_choice = top
        elif nearest == 2:
            winning_choice = bottom
        else:
            winning_choice = right        
            
    cv2.imshow('Webcam', img)
    cv2.imshow('Mask', green_mask_orig)

    print('Left:', left, ';  Right:', right, ';  Top:', top, ';  Bottom:', bottom, ';  Winning', winning_choice)

    if len(winning_choice) > 0:
        overall_winning.add(winning_choice)
        
    
    if cv2.waitKey(1) & 0xFF==ord('q'): # quit when 'q' is pressed
        cap.release()
        break

    
print(overall_winning)

    
cv2.destroyAllWindows() 
cv2.waitKey(1) # normally unnecessary, but it fixes a bug on MacOS where the window doesn't close
