import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep


class face_regognition:
    def __init__(self):
        self.faces = self.get_encoded_faces()


    def get_encoded_faces(self):
        encoded = {}

        for dirpath, dnames, fnames in os.walk("./faces"):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file("faces/" + f)
                    encoding = fr.face_encodings(face)[0]
                    encoded[f.split(".")[0]] = encoding
                    print('test face encoded')

        return encoded


    def unknown_image_encoded(self, img):

        face = fr.load_image_file(img)
        print("trump face loaded")
        encoding = fr.face_encodings(face)[0]
        print("trump face encoded")

        return encoding


    def classify_face(self, im):

        faces_encoded = list(self.faces.values())
        known_face_names = list(self.faces.keys())

        self.img = cv2.imread(im, 1)
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        # img = img[:,:,::-1]

        face_locations = face_recognition.face_locations(self.img)
        unknown_face_encodings = face_recognition.face_encodings(self.img, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(faces_encoded, face_encoding)


            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = known_face_names[best_match_index]

            face_names.append(name)

            self.display_box(face_locations, face_names)
            cv2.imshow('Video', self.img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names

        sleep(10)


    def display_box(self, face_locations, face_names):
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(self.img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(self.img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(self.img, name, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2)


fr = face_regognition()
fr.classify_face(f"2137.jpeg")





