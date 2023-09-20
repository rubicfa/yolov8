'''
Author: 法振尧 fazhenyao2018@163.com
Date: 2023-09-17 16:04:45
LastEditors: 法振尧 fazhenyao2018@163.com
LastEditTime: 2023-09-17 16:15:19
FilePath: /ultralytics/usr_sc/t2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics import YOLO

model=YOLO('/mnt/e/file/ultralytics/usr_sc/yolov8-pose.yaml')

model.train(data='/mnt/e/file/ultralytics/usr_sc/data.yaml',epochs=1,amp=False, batch=1)