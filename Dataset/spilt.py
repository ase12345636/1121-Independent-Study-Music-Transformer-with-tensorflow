import csv
import os
import shutil

with open('maestro-v3.0.0.csv', 'r',encoding="utf-8") as file:
    csv_reader = csv.DictReader(file)
    data = [row for row in csv_reader]

for i in range (len(data)):
    sourcepath = os.path.dirname(os.path.abspath(__file__))
    sourcepath += "\\" + data[i]["midi_filename"]
    dirpath = os.path.dirname(os.path.abspath(__file__))
    dirpath += "\\"
    if data[i]["split"]=="train":
        dirpath += "train"
    elif data[i]["split"]=="validation":
        dirpath += "vaild"
    elif data[i]["split"]=="test":
        dirpath += "test"
    dirpath += "\\"
    shutil.move(sourcepath,dirpath)
