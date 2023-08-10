import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyttsx3
import os

name2=""

# Initializing the Cascade classifier
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
#list of known family members for 
# Load a sample picture.
suhanee = face_recognition.load_image_file("Suhanee_Hingorani.jpeg")
suhanee_face_encoding = face_recognition.face_encodings(suhanee)[0]

obama = face_recognition.load_image_file("President_Barack_Obama.jpeg")
obama_face_encoding = face_recognition.face_encodings(obama)[0]

donald = face_recognition.load_image_file("President_Donald_Trump.jpeg")
donald_face_encoding = face_recognition.face_encodings(donald)[0]

known_face_encodings = [suhanee_face_encoding, obama_face_encoding,donald_face_encoding]
known_face_names = ["Suhanee Hingorani" ,"President Barack Obama","President Donald Trump"]

engine=pyttsx3.init()
sound=engine.getProperty("voices")
engine.setProperty("voice", sound[1].id)
rate=engine.getProperty("rate")
engine.setProperty("rate", 150)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


    
while True:
    #print("Hello world")
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    if cv2.waitKey(20)& 0xFF==ord('q'):
       break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #rgb_small_frame = small_frame[:, :, ::-1]
    # Modified and easy version.
    rgb_small_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     # Initializing the pyttsx3 engine.
    


    # Only process every other frame of video to save time
    
    #if cv2.waitKey(1) & 0xFF==ord('p'):
     #   print("Hi")
    if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        #face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown Person"
           
            #use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                print(name)
                name2=name
                engine.say(f"{name} says hello.")
                engine.runAndWait()
               
           
                           
            # Display the results ==========Modified===================
    faces = faceCascade.detectMultiScale(
    rgb_small_frame,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
    )

     # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
       
        # For displaying the name
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv2.putText(frame,name2, (w, w), font, 1.0, (255, 255, 255), 1)       

        face_names.append(name2)
        process_this_frame = not process_this_frame


    # Display the resulting image
    cv2.imshow('Video', frame)

    '''# Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break'''

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
