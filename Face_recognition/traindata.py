import os,numpy

def trainData():
    # Create a list of images and a list of corresponding names
    print("Training the data....")
    (images, lables, names, ids) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            
            names[ids] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = ids
                images.append(cv2.imread(path, 0))
                lables.append(int(lable))
            ids += 1


    # Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'    
    model.train(images, lables)
    print('Training completed..')
    
    
if __name__ == '__main__':
    trainData()