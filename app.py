

from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
app=Flask(__name__)
camera = cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.
krish_image = face_recognition.load_image_file("Krish/krish.jpg")
krish_face_encoding = face_recognition.face_encodings(krish_image)[0]

# Load a sample picture and learn how to recognize it.
mahesh_image = face_recognition.load_image_file("mahesh/mahesh.png")
mahesh_face_encoding = face_recognition.face_encodings(mahesh_image)[0]

# Load a second sample picture and learn how to recognize it.
bradley_image = face_recognition.load_image_file("Bradley/bradley.jpg")
bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

# Load a sample picture and learn how to recognize it.
jaswanth_image = face_recognition.load_image_file("jaswanth/jaswanth.png")
jaswanth_face_encoding = face_recognition.face_encodings(jaswanth_image)[0]

# Load a sample picture and learn how to recognize it.
rechal_image = face_recognition.load_image_file("rechal/rechal.png")
rechal_face_encoding = face_recognition.face_encodings(rechal_image)[0]

# Load a sample picture and learn how to recognize it.
charishma_image = face_recognition.load_image_file("charishma/charishma.png")
charishma_face_encoding = face_recognition.face_encodings(charishma_image)[0]

# Load a sample picture and learn how to recognize it.
jyothika_image = face_recognition.load_image_file("jyothika/jyothika.png")
jyothika_face_encoding = face_recognition.face_encodings(jyothika_image)[0]

# Load a sample picture and learn how to recognize it.
pujitha_image = face_recognition.load_image_file("pujitha/pujitha.png")
pujitha_face_encoding = face_recognition.face_encodings(pujitha_image)[0]

# Load a sample picture and learn how to recognize it.
divyapavane_image = face_recognition.load_image_file("divyapavane/divyapavane.png")
divyapavane_face_encoding = face_recognition.face_encodings(divyapavane_image)[0]

# Load a sample picture and learn how to recognize it.
abhin_image = face_recognition.load_image_file("abhin/abhin.png")
abhin_face_encoding = face_recognition.face_encodings(abhin_image)[0]

# Load a sample picture and learn how to recognize it.
nithi_image = face_recognition.load_image_file("nithi/nithi.png")
nithi_face_encoding = face_recognition.face_encodings(nithi_image)[0]

# Load a sample picture and learn how to recognize it.
prince_image = face_recognition.load_image_file("prince/prince.png")
prince_face_encoding = face_recognition.face_encodings(prince_image)[0]

# Load a sample picture and learn how to recognize it.
satya_image = face_recognition.load_image_file("satya/satya.png")
satya_face_encoding = face_recognition.face_encodings(satya_image)[0]

# Load a sample picture and learn how to recognize it.
sriram_image = face_recognition.load_image_file("sriram/sriram.png")
sriram_face_encoding = face_recognition.face_encodings(sriram_image)[0]

# Load a sample picture and learn how to recognize it.
gandhi_image = face_recognition.load_image_file("gandhi/gandhi.png")
gandhi_face_encoding = face_recognition.face_encodings(gandhi_image)[0]

# Load a sample picture and learn how to recognize it.
anil_image = face_recognition.load_image_file("anil/anil.png")
anil_face_encoding = face_recognition.face_encodings(anil_image)[0]

# Load a sample picture and learn how to recognize it.
balu_image = face_recognition.load_image_file("balu/balu.png")
balu_face_encoding = face_recognition.face_encodings(balu_image)[0]

# Load a sample picture and learn how to recognize it.
jayakrishna_image = face_recognition.load_image_file("jayakrishna/jayakrishna.png")
jayakrishna_face_encoding = face_recognition.face_encodings(jayakrishna_image)[0]

# Load a sample picture and learn how to recognize it.
anvitha_image = face_recognition.load_image_file("anvitha/anvitha.png")
anvitha_face_encoding = face_recognition.face_encodings(anvitha_image)[0]
# Load a sample picture and learn how to recognize it.
kavya_image = face_recognition.load_image_file("kavya/kavya.png")
kavya_face_encoding = face_recognition.face_encodings(kavya_image)[0]
# Load a sample picture and learn how to recognize it.
prineetha_image = face_recognition.load_image_file("prineetha/prineetha.png")
prineetha_face_encoding = face_recognition.face_encodings(prineetha_image)[0]

# Load a sample picture and learn how to recognize it.
sailaja_image = face_recognition.load_image_file("sailaja/sailaja.png")
sailaja_face_encoding = face_recognition.face_encodings(sailaja_image)[0]






# Create arrays of known face encodings and their names
known_face_encodings = [
    krish_face_encoding,
    mahesh_face_encoding,
    bradley_face_encoding,
    jaswanth_face_encoding,
    rechal_face_encoding,
    charishma_face_encoding,
    jyothika_face_encoding,
    pujitha_face_encoding,
    divyapavane_face_encoding,
    abhin_face_encoding,
    nithi_face_encoding,
    prince_face_encoding,
    satya_face_encoding,
    sriram_face_encoding,
    gandhi_face_encoding,
    anil_face_encoding,
    balu_face_encoding,
    jayakrishna_face_encoding,
    anvitha_face_encoding,
    kavya_face_encoding,
    prineetha_face_encoding,
    sailaja_face_encoding



]
known_face_names = [
    "Krish",
    "mahesh",
    "Bradly",
    "jaswanth",
    "rechal",
    "charishma",
    "jyothika",
    "pujitha",
    "divyapavane",
    "abhin",
    "nithi",
    "prince",
    "satya",
    "sriram",
    "gandhi",
    "anil",
    "balu",
    "jayakrishna",
    "anvitha",
    "kavya",
    "praneetha",
    "sailaja"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)