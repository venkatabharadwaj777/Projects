import os, cv2

def recognizeFace():
    (images, lables, names, ids) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            
            names[ids] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = ids
                #images.append(cv2.imread(path, 0))
                #lables.append(int(lable))
            ids += 1
            
    isRecognised=False
    vidcap = cv2.VideoCapture('videos/input.avi')
    if (vidcap.isOpened()== False): 
      print("Error opening video  file")
      return False
    
    while(vidcap.isOpened()):
          # Capture frame-by-frame     
        
        success, im = vidcap.read()
        if success == True:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                # Try to recognize the face
                prediction = model.predict(face_resize)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
                if prediction[1]<100:  
                   isRecognised = True
                   print(names[prediction[0]])
                   print(prediction)
                   return isRecognised  
        else: 
            break

        # When everything done, release 
        # the video capture object
    vidcap.release()

     # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognizeFace()