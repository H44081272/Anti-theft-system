#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
Detect Objects Using Your Webcam
================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" detection model to
# detect objects in the video stream extracted from your camera.

# %%
# Create the data directory
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The snippet shown below will create the ``data`` directory where all our data will be stored. The
# code will create a directory structure as shown bellow:
#
# .. code-block:: bash
#
#     data
#     └── models
#
# where the ``models`` folder will will contain the downloaded models.
# %%
# Download the model
# ~~~~~~~~~~~~~~~~~~
# The code snippet shown below is used to download the object detection model checkpoint file,
# as well as the labels file (.pbtxt) which contains a list of strings used to add the correct
# label to each detection (e.g. person).
#
# The particular detection algorithm we will use is the `SSD ResNet101 V1 FPN 640x640`. More
# models can be found in the `TensorFlow 2 Detection Model Zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>`_.
# To use a different model you will need the URL name of the specific model. This can be done as
# follows:
#
# 1. Right click on the `Model name` of the model you would like to use;
# 2. Click on `Copy link address` to copy the download link of the model;
# 3. Paste the link in a text editor of your choice. You should observe a link similar to ``download.tensorflow.org/models/object_detection/tf2/YYYYYYYY/XXXXXXXXX.tar.gz``;
# 4. Copy the ``XXXXXXXXX`` part of the link and use it to replace the value of the ``MODEL_NAME`` variable in the code shown below;
# 5. Copy the ``YYYYYYYY`` part of the link and use it to replace the value of the ``MODEL_DATE`` variable in the code shown below.
#
# For example, the download link for the model used below is: ``download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz``

# 引入套件
import tkinter as tk
import tkinter.messagebox
#import abcde
#建立應用程式視窗
window = tk.Tk()
top_frame = tk.Frame(window)
window.title("居家防盜系統")
window.geometry("300x200")  # width X height
label = tk.Label(window,text = '歡迎使用居家防盜系統')  # 顯示文字
label.pack()

def high_protection():
    
    import os
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    for dir in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    import tarfile
    import urllib.request

    
    # Download and extract model
    MODEL_DATE = '20200711'
    MODEL_NAME = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
    MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
    MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
    MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
    PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
    PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
    PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
    if not os.path.exists(PATH_TO_CKPT):
        print('Downloading model. This may take a while... ', end='')
        urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
        tar_file = tarfile.open(PATH_TO_MODEL_TAR)
        tar_file.extractall(MODELS_DIR)
        tar_file.close()
        os.remove(PATH_TO_MODEL_TAR)
        print('Done')

    # Download labels file
    LABEL_FILENAME = 'mscoco_label_map.pbtxt'
    LABELS_DOWNLOAD_BASE =         'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
    if not os.path.exists(PATH_TO_LABELS):
        print('Downloading label file... ', end='')
        urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
        print('Done')

    # %%
    # Load the model
    # ~~~~~~~~~~~~~~
    # Next we load the downloaded model

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import config_util
    #from object_detection.utils import visualization_utils as viz_utils
    from object_detection.utils import visualization_utils as vu
    from object_detection.builders import model_builder
    import winsound

    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    def get_classes_name_and_scores(
        boxes,
        classes,
        scores,
        category_index,
        max_boxes_to_draw=20,
        min_score_thresh=.9): # returns bigger than 90% precision
        display_str = {}
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                if classes[i] in six.viewkeys(category_index):
                    display_str['name'] = category_index[classes[i]]['name']
                    display_str['score'] = '{}%'.format(int(100 * scores[i]))

        return display_str

    # %%
    # Load label map data (for plotting)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Label maps correspond index numbers to category names, so that when our convolution network
    # predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
    # functions, but anything that returns a dictionary mapping integers to appropriate string labels
    # would be fine.
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    # %%
    # Define the video stream
    # ~~~~~~~~~~~~~~~~~~~~~~~
    # We will use `OpenCV <https://pypi.org/project/opencv-python/>`_ to capture the video stream
    # generated by our webcam. For more information you can refer to the `OpenCV-Python Tutorials <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#capture-video-from-camera>`_
    import cv2
    cap = cv2.VideoCapture(0)
    import pickle
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from pathlib import Path
    
    
    def get_photo():
        #cap=cv2.VideoCapture(0)
        #f,frame=cap.read()
        
        cv2.imwrite('image.jpg',image_np) #根據指定的格式將圖像保存在當前工作目錄中。
        
        #cap.release()

    def waring_message():
        
        content = MIMEMultipart()  # 建立MIMEMultipart物件
        content["subject"] = "!Warning!"  # 郵件標題
        content["from"] = "h44081272@gs.ncku.edu.tw"  # 寄件者
        content["to"] = "h44081272@gs.ncku.edu.tw"  # 收件者
        content.attach(MIMEText("Stranger is coming!"))  # 郵件純文字內容
        content.attach(MIMEImage(Path("image.jpg").read_bytes()))  # 郵件圖片內容

        server=smtplib.SMTP_SSL('smtp.gmail.com',465)
        server.ehlo()#回應傳送伺服器的指令
        server.login("h44081272@gs.ncku.edu.tw","Daisy0302")
        server.send_message(content)
        server.quit()

        print('Stranger is coming!')
    def Serious_waring():
        
        content = MIMEMultipart()  # 建立MIMEMultipart物件
        content["subject"] = "!!!Serious_waring!!!"  # 郵件標題
        content["from"] = "h44081272@gs.ncku.edu.tw"  # 寄件者
        content["to"] = "h44081272@gs.ncku.edu.tw"  # 收件者
        content.attach(MIMEText("Someone carrying a weapon is coming!!!"))  # 郵件純文字內容
        content.attach(MIMEImage(Path("image.jpg").read_bytes()))  # 郵件圖片內容
        
        server=smtplib.SMTP_SSL('smtp.gmail.com',465)
        server.ehlo()#回應傳送伺服器的指令
        server.login("h44081272@gs.ncku.edu.tw","Daisy0302")
        server.send_message(content)
        server.quit()

        print('Someone carrying a weapon is coming!!!')

# %%
# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# Here are some simple things to try out if you are curious:
#
# * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.

        
    import numpy as np
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    font = cv2.FONT_HERSHEY_SIMPLEX

 
    while True:
        # Read frame from camera
        t1 = cv2.getTickCount()
        ret, image_np = cap.read()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        label  = vu.visualize_boxes_and_labels_on_image_array(
                  image_np_with_detections,
                  detections['detection_boxes'][0].numpy(),
                  (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                  detections['detection_scores'][0].numpy(),
                  category_index,
                  use_normalized_coordinates=True,
                  max_boxes_to_draw=8,
                  min_score_thresh=.50,
                  agnostic_mode=False)
        
        objects1=[]
        for index, value in enumerate(detections['detection_classes'][0].numpy()):
            try:
                object_dict = {}
                
                if detections['detection_scores'][0,index].numpy() >0.4:
                    value1 = value+1
                    object_dict[(category_index.get(value1)).get('name')] =                                 detections['detection_scores'][0, index].numpy()
                    n = 'scissors'
                    n1= 'knife'
                    n2= 'person'
                    x = (category_index.get(value1).get('name'))
                    objects1.append(x)
                    print(objects1)
                    
                    if n in objects1 and n2 in objects1:
                        winsound.Beep(1000,2000)
                        get_photo()
                        Serious_waring()
                        
                    elif n1 in objects1 and n2 in objects1:                           
                        winsound.Beep(1000,2000)
                        get_photo()
                        Serious_waring()
                        
                    elif n2 in objects1 :                           
                        get_photo()
                        waring_message()

                            
                    
                                            



            except:
                pass
        
        #test()          
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        
        cv2.putText(image_np_with_detections,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        # Display output
        #cv2.imshow('object detection', image_np_with_detections)
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()

def low_protection():
    newWindow = tk.Toplevel(window)
    newWindow.geometry("200x100")
    varName = tk.StringVar()
    varName.set('')
    varPwd = tk.StringVar()
    varPwd.set('')
    #建立標籤
    labelName = tk.Label(newWindow, text='使用者名稱:', justify=tk.RIGHT, width=80)
    #將標籤放到視窗上
    labelName.place(x=10, y=5, width=80, height=20)
    #建立文字框，同時設定關聯的變數
    entryName = tk.Entry(newWindow, width=80,textvariable=varName)
    entryName.place(x=100, y=5, width=80, height=20)
    labelPwd = tk.Label(newWindow, text='密 碼:', justify=tk.RIGHT, width=80)
    labelPwd.place(x=10, y=30, width=80, height=20)
    #建立密碼文字框
    entryPwd = tk.Entry(newWindow, show='*',width=80, textvariable=varPwd)
    entryPwd.place(x=100, y=30, width=80, height=20)
    #登入按鈕事件處理函式
    def login():
        #獲取使用者名稱和密碼
        name = entryName.get()
        pwd = entryPwd.get()
        if name=='h44081272'and pwd=='123456':
            tkinter.messagebox.showinfo(title='居家防盜系統',message='登入成功！')
            button3['state'] = tk.DISABLED
            
        else:
            tkinter.messagebox.showerror(title='居家防盜系統', message='登入失敗')
    #建立按鈕元件，同時設定按鈕事件處理函式
    buttonOk = tk.Button(newWindow, text='登入', command=login)
    buttonOk.place(x=30, y=70, width=50, height=20)
    #取消按鈕的事件處理函式
    def cancel():
        #清空使用者輸入的使用者名稱和密碼
        varName.set('')
        varPwd.set('')
    buttonCancel = tk.Button(newWindow, text='取消', command=cancel)
    buttonCancel.place(x=90, y=70, width=50, height=20)

button1 = tk.Button(window,text="高度保護",command=high_protection)
button1.pack()
button2 = tk.Button(window,text="低度保護",command=low_protection)
button2.pack()
button3 = tk.Button(window,text="離開",command=window.destroy)
button3.pack()


#啟動訊息迴圈
window.mainloop()

