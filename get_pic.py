#用摄像头捕获视频
import os

import cv2
'''
创建VideoCapture对象，它的参数可以是设备的索引号，或者一个文件。
设备索引号就是在指定要使用的摄像头。
一般的笔记本电脑都有内置摄像头。所以参数就是 0。
你可以通过设置成 1 或者其他的来选择别的摄像头。
'''
cap_width = 640
cap_height = 480
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
picNum = 0

while(True):
    # Capture frame-by-frame
    '''
ret, frame = cap.read()  cap = cv2.VideoCapture(0)
返回值含义：
    参数ret 为True 或者False,代表有没有读取到图片
    第二个参数frame表示截取到一帧的图片
其他情况说明：
    有时 cap 可能不能成功的初始化摄像头设备。这种情况下上面的代码会报
错。你可以使用 cap.isOpened()，来检查是否成功初始化了。如果返回值是
True，那就没有问题。否则就要使用函数 cap.open()。
    其中的一些值可以使用 cap.set(propId,value) 来修改，value 就是你想要设置成的新值。
    例如，我可以使用 cap.get(3) 和 cap.get(4) 来查看每一帧的宽和高。默认情况下得到的值是 640X480。但是我可以使用 ret=cap.set(3,320)和 ret=cap.set(4,240) 来把宽和高改成 320X240。
    '''
    ret, frame = cap.read()
    # image_read = cv2.imread("1.jpg")
    # print(type(frame))
    # print(type(image_read))
    # Our operations on the frame come here

    '''
    cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
    cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
    cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
    '''
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    # 显示图片参数，第一个参数是窗口的名字，其次才是我们的图像
    # cv2.imshow('image',img)
    # cv2.imshow('1',img)
    image = cv2.flip(frame, 1)
    cv2.imshow('frame',image)
    className = "left"
    path = 'C:\\Users\\10570\\Downloads\\hand2\\test\\'+ className+'\\'




    if  cv2.waitKey(1) & 0xFF == ord('k'):
        picNum+=1
        cv2.imwrite(path+(str(picNum)+'.jpg'), image)
        print(picNum)


    #等候1ms,播放下一帧，或者按q键退出
    if  cv2.waitKey(1) & 0xFF == ord('q'):
         break
# When everything done, release the capture
#用来停止捕获视频
cap.release()
#关闭相应的显示窗口的
cv2.destroyAllWindows()