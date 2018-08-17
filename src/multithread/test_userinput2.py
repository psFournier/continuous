import threading
import time

kbdInput = ''
playingID = ''
finished = True



while True:
    print("kbdInput: {}".format(kbdInput))
    print("playingID: {}".format(playingID))
    if playingID != kbdInput:
        print("Received new keyboard Input. Setting playing ID to keyboard input value")
        playingID = kbdInput
    else:
        print("No input from keyboard detected. Sleeping 2 seconds")
    if finished:
        finished = False
        listener = threading.Thread(target=kbdListener)
        listener.start()
    time.sleep(2)