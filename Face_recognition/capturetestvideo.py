import cv2

def captureTestVideo():
    # Part 1: Create fisherRecognizer
    print('Recognizing Face Please Be in sufficient Lights...')
    #Capture video from webcam
    vid_capture = cv2.VideoCapture(0)
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))

    size = (frame_width, frame_height)
    output = cv2.VideoWriter('videos/input.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, size)
    while(True):
         # Capture each frame of webcam video
         ret,frame = vid_capture.read()
         cv2.imshow("My cam video", frame)
         output.write(frame)
         # Close and break the loop after pressing "x" key
         if cv2.waitKey(1) &0XFF == ord('x'):
             break
    # close the already opened camera
    vid_capture.release()
    # close the already opened file
    output.release()
    # close the window and de-allocate any associated memory usage
    cv2.destroyAllWindows()


if __name__ == '__main__':
    captureTestVideo()