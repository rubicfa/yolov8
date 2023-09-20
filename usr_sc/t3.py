'''
Author: 法振尧 fazhenyao2018@163.com
Date: 2023-09-20 22:40:44
LastEditors: 法振尧 fazhenyao2018@163.com
LastEditTime: 2023-09-20 22:46:56
FilePath: /ultralytics/usr_sc/t3.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics import YOLO
model=YOLO('yolov8l.pt')
file="/mnt/e/file/ultralytics/usr_sc/data/faces/299733036_fff5ea6f8e.jpg"

model.predict(source=file,save=True,show=True)
