import cv2
import mediapipe as mp
import numpy as np
from scipy import stats

import facemeshANN as classifier
import preprocessFERplus as preprocess

from paho.mqtt import client as mqtt_client

import json

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

converter = preprocess.prepareforANN
model = classifier.ANNClassifier(input_size=478 * 3,
                                 output_size=7,
                                 dropout=0.5)
model = classifier.getmodel(model, './model/FERplusmeshANNColab.pt')

broker = '192.168.64.132'
port = 8765
topic = "North/facemesh"
# generate client ID with pub prefix randomly
client_id = f'facemesh-pub'

FERclassName = [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'None'
]


def drawing(drawer, image, landmark_list, connections, landmark_drawing_spec,
            connection_drawing_spec):
    
    drawer.draw_landmarks(image=image,
                          landmark_list=landmark_list,
                          connections=connections,
                          landmark_drawing_spec=landmark_drawing_spec,
                          connection_drawing_spec=connection_drawing_spec)
    return image


def labeling(image, landmark_list, emotions, num):
    return cv2.putText(image,
                       f'{num}:{FERclassName[emotions.argmax(1).item()]}',
                       (int(landmark_list[10].x * image.shape[1]),
                        int(landmark_list[10].y * image.shape[0])),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                       cv2.LINE_AA)
    
def emo_to_dict(d, emotions):
    d = {}
    for i in range(len(emotions)):
        d[f'face{i}'] = FERclassName[int(emotions[i])]
    return d
    
def connect_mqtt():

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

client = connect_mqtt()


def publish(client, topic, msg):
    result = client.publish(topic, msg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
frame_count = 0
max_num_faces = 4
n_frames = 10
results_mat = np.ones((max_num_faces, n_frames)) * 7

with mp_face_mesh.FaceMesh(max_num_faces=max_num_faces,
                           refine_landmarks=True,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.6) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                emotions = classifier.predict(
                    model, np.array([converter(face_landmarks.landmark)]))
                # print(f'face {i}:{FERclassName[emotions.argmax(1).item()]}')
                results_mat[i, frame_count] = emotions.argmax(1).item()
                # print(f'face {i}:{results_mat[i]}')
                image = drawing(
                    mp_drawing, image, face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION, None,
                    mp_drawing_styles.get_default_face_mesh_tesselation_style(
                    ))
                image = drawing(
                    mp_drawing, image, face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS, None,
                    mp_drawing_styles.get_default_face_mesh_contours_style())
                image = drawing(
                    mp_drawing, image, face_landmarks,
                    mp_face_mesh.FACEMESH_IRISES, None,
                    mp_drawing_styles.
                    get_default_face_mesh_iris_connections_style())
                image = labeling(image, face_landmarks.landmark, emotions, i)
        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        cv2.imshow('MediaPipe Face Mesh', image)
        frame_count += 1
        if frame_count == n_frames:
            emo = stats.mode(results_mat, axis=1)[0].flatten().tolist()
            msg = json.dumps(emo_to_dict({}, emo))
            print(msg)
            frame_count = 0
            results_mat = np.ones((max_num_faces, n_frames)) * 7
            publish(client, topic, msg)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()