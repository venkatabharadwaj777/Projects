import cv2, sys, numpy, os, time

datasets = 'datasets'  
(width, height) = (130, 100)  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create() 


def captureNewFace():
    
    print("Your entering frame your ready to give pictures press: q ")

    webcam = cv2.VideoCapture(0)
    
    while(webcam.isOpened()):
            ret, frame = webcam.read()

            if ret == True:
                # Display the resulting frame
                cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    # When everything done, release the video capture object
    webcam.release()

    cv2.destroyAllWindows()
    
def takepics():
    
    sub_data = input("Enter your name: ")
    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    webcam = cv2.VideoCapture(0)
    
    
    try:
        
        print("started capture face....")
        
        count = 1
        
        # The program loops until it has 50 images of the face.    

        while count < 50:

            ret, frame  = webcam.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      

            faces = face_cascade.detectMultiScale(gray, 1.3, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('% s/% s.png' % (path, count), face_resize)

            count += 1

            cv2.imshow('OpenCV', gray)
            
            # sleep time 0.30sec(10 pics in 3 sec)
            # sleep time 0.50sec(10 pics in 5 sec )
            time.sleep(0.30)
            
        print("Completed....")
                

    except Exception as e:
        print("An exception occurred: ",e,sys.exc_info()[0])
    finally:
        webcam.release()
        cv2.destroyAllWindows()
        
        
        
if __name__ == '__main__':
    captureNewFace()
    takepics()