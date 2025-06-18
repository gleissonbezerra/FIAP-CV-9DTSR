# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import cv2 as cv
import requests

import time



def resize(img, scale_percent):

    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)

    dim = (width, height)

    return cv.resize(img, dim, interpolation = cv.INTER_AREA)


def processFrame(frameBytes):
    global INFERENCE_URL

    multipart_form_data = {'frame': ("frame.jpg", frameBytes)}

    response = requests.post(INFERENCE_URL,files=multipart_form_data, timeout=2)

    return response.json()


def main():
    global vf
    global INFERENCE_URL

    print ( "Camera Module for Python" )

    CAMERA_INDEX = 1
    CAMERA_INTERVAL = 0.1
    INFERENCE_URL = "http://localhost:8080/analyze" 

    vf = cv.VideoCapture(CAMERA_INDEX)

    if not vf.isOpened():
        print("Error opening camera!!!")
        exit()


    try:

        while True:

        
            (ret, frame) = vf.read()

            if not ret:
                print("No frame data!!!")
                continue

            #frame = resize(frame, 0.5)

            frameBytes = cv.imencode( '.jpg', frame)[1].tobytes()
            r = processFrame(frameBytes)

            print (r)

            time.sleep(CAMERA_INTERVAL)
            

    except Exception as e:
        print("Unexpected error %s " % e)
        raise
    finally:
        print("Shutting down Camera Modulre...")
        vf.release()

if __name__ == "__main__":
    main()
