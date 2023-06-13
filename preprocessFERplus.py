import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

tqdm.pandas()


class FERdata(object):

    def __init__(self, path_to_csv):
        self.original_df = pd.read_csv(path_to_csv)
        self.original_df['img'] = self.original_df.progress_apply(to_img,
                                                                  axis=1)

    def get_df(self,
               mode='ANN',
               drawlandmarks=False,
               mapping=False,
               cmap='GRAY',
               sample=False,
               sample_size=10):

        if sample == True:
            return getlandmark(
                pd.concat([
                    self.original_df[self.original_df[' Usage'] ==
                                     'Training'][:sample_size],
                    self.original_df[self.original_df[' Usage'] ==
                                     'PublicTest'][:sample_size],
                    self.original_df[self.original_df[' Usage'] ==
                                     'PrivateTest'][:sample_size]
                ]), mode, drawlandmarks, mapping, cmap)
        else:
            return getlandmark(self.original_df, mode, drawlandmarks, mapping,
                               cmap)


def to_img(row):
    return np.array(row[' pixels'].split(' ')).reshape(48, 48).astype('uint8')


def prepareforANN(lanmarks):
    return np.array([[landmark.x, landmark.y, landmark.z]
                     for landmark in lanmarks])


def drawalllandmark(annotated_image, result):
    for face_landmarks in result.multi_face_landmarks:
        # print('face_landmarks:', face_landmarks)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.
            get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(image=annotated_image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.
                                  get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.
            get_default_face_mesh_iris_connections_style())

    return annotated_image


def emotionmapping(emotion):
    if emotion == 0:
        return 'anger'
    elif emotion == 1:
        return 'disgust'
    elif emotion == 2:
        return 'fear'
    elif emotion == 3:
        return 'happy'
    elif emotion == 4:
        return 'sad'
    elif emotion == 5:
        return 'surprise'
    elif emotion == 6:
        return 'neutral'


def getlandmark(df, mode, draw, map, cmap):
    new_df = pd.DataFrame(columns=['usage', 'feature', 'target'])
    draw_img = []
    usage_map = {
        'Training': 'train',
        'PublicTest': 'val',
        'PrivateTest': 'test',
    }
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        for data in tqdm(df[[' Usage', 'img', 'emotion']].values):
            # Convert the BGR image to RGB before processing.
            if cmap == 'GRAY':
                img = cv2.cvtColor(data[1], cv2.COLOR_GRAY2RGB)
            elif cmap == 'BGR':
                img = cv2.cvtColor(data[1], cv2.COLOR_BGR2RGB)
            else:
                img = data[1]

            if map == True:
                print('Currently have only not map')
            else:
                emotion = data[2]

            if mode == 'ANN':
                result = face_mesh.process(img)
                if not result.multi_face_landmarks:
                    continue

                np_landmark = prepareforANN(
                    result.multi_face_landmarks[0].landmark)

                if draw == True:
                    annotated_image = cv2.cvtColor(data[1],
                                                   cv2.COLOR_GRAY2RGB).copy()
                    annotated_image = drawalllandmark(annotated_image, result)
                    draw_img.append(annotated_image)

                new_df = pd.concat([
                    new_df,
                    pd.DataFrame([[usage_map[data[0]], np_landmark, emotion]],
                                 columns=['usage', 'feature', 'target'])
                ],
                                   ignore_index=True)

            if mode == 'IMGANN':
                new_df = pd.concat([
                    new_df,
                    pd.DataFrame([[
                        usage_map[data[0]],
                        cv2.cvtColor(data[1], cv2.COLOR_GRAY2RGB), emotion
                    ]],
                                 columns=['usage', 'feature', 'target'])
                ],
                                   ignore_index=True)

            else:
                print('Currently have only ANN, IMGANN mode')
                break

        if draw is True:
            new_df['draw_img'] = draw_img

    return new_df