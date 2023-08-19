import cv2
import mediapipe as mp
import numpy as np
from scipy import stats
import time

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
                                 dropout=0.1)
model = classifier.getmodel(model, './model/FERplusmeshANNColabRotate50.pt')

broker = '192.168.64.132'
port = 1883
topic = "facial_exp"
# generate client ID with pub prefix randomly
client_id = f'facemesh-pub'

FERclassName = [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'None'
]

RussellclassName = {
    'Happy': 'Happy',
    'Surprise': 'Happy',
    'Angry': 'Anger',
    'Fear': 'Anger',
    'Disgust': 'Anger',
    'Sad': 'Sadness',
    'Neutral': 'Neutral',
    'None': 'None'
}


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


def emo_to_dict(emotions,
                confident=None,
                face_sizes=None,
                russell=False,
                d=dict()):
    for i in range(d['num_faces']):
        face = {}
        if russell is True:
            face['emotion'] = RussellclassName[FERclassName[int(emotions[i])]]
            face['pred_emo'] = FERclassName[int(emotions[i])]
        else:
            face['emotion'] = FERclassName[int(emotions[i])]
        
        if confident is not None:
            face['confident'] = round(confident[i] * 100, 4)
        if face_sizes is not None:
            face['face_size'] = round(face_sizes[i], 4)
        d[f'face{i}'] = face

    # print(d)
    return d


def face_size_cals(landmarks):
    w = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
    h = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
    # print(w*h)
    return round(w * h, 4)


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
num_faces = 0
max_num_faces = 2
n_frames = 10
results_mat = np.ones((max_num_faces, n_frames)) * 7
face_sizes = np.zeros(max_num_faces)
conf = np.zeros(max_num_faces)
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

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
        new_frame_time = time.time()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)
            for i, face_landmarks in enumerate(results.multi_face_landmarks):
                emotions = classifier.predict(
                    model, np.array([converter(face_landmarks.landmark)]))
                # print(f'face {i}:{FERclassName[emotions.argmax(1).item()]}')
                results_mat[i, frame_count] = emotions.argmax(1).item()
                conf[i] = emotions.max()
                face_sizes[i] = face_size_cals(
                    np.array(converter(face_landmarks.landmark)))
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
        times_delay = new_frame_time-prev_frame_time
        fps = 1/(times_delay)
        prev_frame_time = new_frame_time
        print(f'fps: {fps}, time: {times_delay}')
        frame_count += 1
        if frame_count == n_frames:
            emo = stats.mode(results_mat, axis=1)[0].flatten().tolist()
            d = {"name": "facemesh data", 'num_faces': num_faces}
            msg = json.dumps(emo_to_dict(emo, conf, face_sizes, True,
                                         d))  # with confident face size and mapped emotion
            # msg = json.dumps(emo_to_dict(emotions=emo, d=d)) # without confident and face size
            # print(msg)
            frame_count = 0
            results_mat = np.ones((max_num_faces, n_frames)) * 7
            face_sizes = np.zeros(max_num_faces)
            conf = np.zeros(max_num_faces)
            publish(client, topic, msg)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()