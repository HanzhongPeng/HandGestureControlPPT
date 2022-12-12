#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import math
import time
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


import pyautogui
import threading

import win32gui
import win32con

imageCount = 0
is_control = False
joint_list = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]  # 手指关节序列

def cal_num(num):
    return num+1


def window_top():
    hwnd = win32gui.FindWindow(None, "Hand Gesture Recognition")
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER | win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)


def main():
    # t1 = threading.Thread(target=hand_recognition,name='1')
    # t1.start()
    # t1.join()
    keypoint_classifier = KeyPointClassifier()
    hand_recognition('enter',57,keypoint_classifier)


def get_straight_finger_list(RHL):
    fingerlist = [0,0,0,0,0]
    w = 640
    h = 480
    three = (int(RHL.landmark[3].x * w), int(RHL.landmark[3].y * h))

    zero = (int(RHL.landmark[0].x * w), int(RHL.landmark[0].y * h))

    # 勾股定理
    length_compare = int(three[0] - zero[0]) ** 2 + int(three[1] - zero[1]) ** 2
    length_compare = int(math.sqrt(length_compare))

    thumb_tip = (int(RHL.landmark[4].x * w), int(RHL.landmark[4].y * h))

    length_damuzhi = int(thumb_tip[0] - zero[0]) ** 2 + int(thumb_tip[1] - zero[1]) ** 2
    length_damuzhi = int(math.sqrt(length_damuzhi))

    # 弯曲的话就往列表里记录0
    if length_damuzhi < length_compare:
        fingerlist[0] = 0
        # print("大拇指弯曲")
    else:
        fingerlist[0] = 1
        # print("大拇指直")
    # 计算角度
    for index, joint in enumerate(joint_list):
        a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
        b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
        c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
        # 计算弧度
        radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

        if angle > 180.0:
            angle = 360 - angle

        if angle > 170:
            fingerlist[index + 1] = 1
        else:
            fingerlist[index + 1] = 0


    return fingerlist

def hand_recognition(className,picNum,keypoint_classifier):

    cap_device = 1
    cap_width = 640
    cap_height = 480
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5

    use_static_image_mode = 'store_true'

    use_brect = True

    # Camera preparation ###############################################################


    no_hand_list = []

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )



    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]


    mode = 0



    count = [0,0,0,0,0,0,0,0]
    key_list = ["enter","right","left","+","-","up","down"]
    countThresh = 15
    hand_sign_list = ['Enter','Right','Left','ZoomIn','ZoomOut','Up','Down','Control']


    classNum = 8
    path = 'C:\\Users\\10570\\Downloads\\hand2\\test\\'+ className+'\\'

    fileName = path+ str(picNum)+'.jpg'
    # print(fileName)

    image = cv.imread(fileName)
    # image = cv.flip(image, 1)  # Mirror display
    # cv.imshow("1111",image)

    debug_image = copy.deepcopy(image)

    # Detection implementation #############################################################
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # image.flags.writeable = False
    results = hands.process(image)
    # print(results)


    #  ####################################################################
    if results.multi_hand_landmarks is not None:






        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
            # print(11111)



            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)



            # Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)[0]
            hand_sign_prob = keypoint_classifier(pre_processed_landmark_list)[1]

            if handedness.classification[0].label[0:] == "Left":
                print(11111)
                pass

            # other no_finger_detection
            if handedness.classification[0].label[0:] == "Right":
                print(fileName + '   id:' + str(hand_sign_id) + '   ' + str(hand_sign_prob))
                if hand_sign_prob > 0.95:
                    return hand_sign_id
                else:
                    return str(picNum)+'   '+hand_sign_list[hand_sign_id]

            # finger detection
            # if handedness.classification[0].label[0:] == "Right":
            #     straight_finger_list = get_straight_finger_list(hand_landmarks)
            #     finger_num = sum(straight_finger_list)
            #     print(fileName + '   finger:'+str(finger_num)+'   id:' + str(hand_sign_id) + '   ' + str(hand_sign_prob))
            #     print(fileName + '   id:' + str(hand_sign_id) + '   ' + str(hand_sign_prob))
            #     if finger_num>0 and finger_num<4:
            #         if hand_sign_prob>0.95:
            #             return hand_sign_id
            #         else:
            #             return str(picNum)+'   '+hand_sign_list[hand_sign_id]+'  '+str(hand_sign_prob)
            #     else:
            #         return str(picNum)+'   finger number wrong:'+str(finger_num)


            #other finger detection
            # if handedness.classification[0].label[0:] == "Right":
            #     straight_finger_list = get_straight_finger_list(hand_landmarks)
            #     finger_num = sum(straight_finger_list)
            #     print(fileName + '   finger:'+str(finger_num)+'   id:' + str(hand_sign_id) + '   ' + str(hand_sign_prob))
            #     print(fileName + '   id:' + str(hand_sign_id) + '   ' + str(hand_sign_prob))
            #     if finger_num>0 and finger_num<4:
            #         if hand_sign_prob>0.95:
            #             return str(picNum)+'   '+hand_sign_list[hand_sign_id]
            #         else:
            #             return picNum
            #     else:
            #         return picNum

            # other no_finger_detection
            # if handedness.classification[0].label[0:] == "Right":
            #     print(fileName + '   id:' + str(hand_sign_id) + '   ' + str(hand_sign_prob))
            #     if hand_sign_prob > 0.95:
            #         return str(picNum)+'   '+hand_sign_list[hand_sign_id]
            #     else:
            #         return picNum



                #         count[7] += 1
                #         if count[7] == countThresh:
                #             print(is_control)
                #             count = clear_count(count)
                #             break


                # for i in range(7):
                #     if hand_sign_id == i:
                #         if sum(straight_finger_list)<=4 and sum(straight_finger_list)>0:
                #             count[i] += 1
                #             if count[i] == countThresh:
                #                 press_key(key_list[i])
                #                 print(key_list[i])
                #                 count = clear_count(count)
                #                 break
                #
                #             debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                #             mpDraw = mp.solutions.drawing_utils
                #             mpDraw.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                #             # debug_image = draw_landmarks(debug_image, landmark_list)
                #             debug_image = draw_info_text(
                #                 debug_image,
                #                 brect,
                #                 handedness,
                #                 keypoint_classifier_labels[hand_sign_id]+str(hand_sign_prob)
                #                 )

    else:
        print(path+"  no hand detected")
        return str(picNum)+"  no hand detected"








        # Screen reflection #############################################################



def clear_count(count):
    for i in range(8):
        count[i] = 0
    return count

def press_key(key):
    pyautogui.press(key)

def move_mouse():
    global x_cord
    global y_cord
    global is_move
    while True:
        if is_move==1:
            pyautogui.moveTo(x_cord * 4, y_cord * 3)


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list



def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            global imageCount
            imageCount += 1
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

    return



def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   ):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    # info_text = handedness.classification[0].label[0:]
    # if hand_sign_text != "":
        # info_text = info_text + ':' + hand_sign_text
    info_text= hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    cv.putText(image, hand_sign_text, (180, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)


    return image



def draw_info(image, fps, mode, number):
    global imageCount
    global is_control

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number)+"  第"+str(imageCount)+"张", (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
