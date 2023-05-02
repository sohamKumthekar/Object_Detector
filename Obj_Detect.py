import cv2

# img = cv2.imread('lena.jpg' , 1)
cap = cv2.VideoCapture(0)
cap.set(3,510)
cap.set(4,510)
thresh = 0.5

classNames = []
classFile = 'coco.names'          #all names in coco.names file are 'class names'
                                  #each class name has classID from 1 to 93 here

with open(classFile , 'rt') as f:
    classNames = f.read().rstrip('\n').split()    #creates list of all class names

# print(classNames)
# print(len(classNames))  #93
 
#import files for model SSD MobileNet
configPath = 'ssd_mobilenet_v3_large_coco.pbtxt'     #models architecture and configuration
weightPath = 'frozen_inference_graph.pb'             #binary representation of model(weight file)

while True:
    ret,img = cap.read()
    if ret == True:
        #set the image for processing
        net = cv2.dnn_DetectionModel(weightPath , configPath)     #load pre-trained model from weight and config files
        net.setInputSize(320,320)                                 #sets size of image before processing(size expected by model)
        net.setInputScale(1/127.5)                                #sets scale factor=1/127.5 to reduce size of captured image
        net.setInputMean((127.5 , 127.5 , 127.5))                 #to set mean values to normalize the i/p images
        net.setInputSwapRB(True)                                  #swap B and G channels to make img BGR 
        
        #ID of class(1-93 here) , confidence value 0-1 , bounding box coordinates
        classIDs , confs , bbox = net.detect(img , confThreshold=thresh) 
        print(classIDs  , bbox)
        #this will have IDs confs bbox for each object detected from the image
        #to iterate through all objects detected

        if len(classIDs) != 0:
            for classID,confidence,box in zip(classIDs.flatten() , confs.flatten() , bbox):    # .flatten() creates list of all classIDs of all objects detected from image
                cv2.rectangle(img , box , (0,255,0) , 3)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(img , classNames[classID - 1].upper() , (box[0]+10 , box[1]+30) , font , 1 , (0,255,0) , 2 , cv2.LINE_AA)
                cv2.putText(img , str(round(confidence*100 , 2)) , (box[0]+200 , box[1]+30) , font , 1 , (0,255,0) , 2 , cv2.LINE_AA)
            
        cv2.imshow('Output' , img)
        k = cv2.waitKey(1)
        if k == 27:
            break
cv2.destroyAllWindows()
