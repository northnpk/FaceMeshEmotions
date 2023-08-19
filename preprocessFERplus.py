import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import mediapipe as mp
from scipy.spatial.transform import Rotation as R

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

tqdm.pandas()

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
    'Neutral': 'Relax',
    'None': 'None'
}

class FERdata(object):

    def __init__(self, path_to_csv):
        print(f'reading csvfile from {path_to_csv}')
        self.df = pd.read_csv(path_to_csv).sample(frac=1, ignore_index=True)
        self.len = len(self.df)
        self.class_name = FERclassName

    def __len__(self):
        return self.len

    def save_df(self, path_to_save):
        self.df.to_csv(path_to_save, index=False)
        print(f'save {path_to_save} done')

    def balance_df(self, mode):
        return pd.concat([
            balance_each_usage(self.df[self.df['usage'] == 'train'], mode),
            balance_each_usage(self.df[self.df['usage'] == 'val'], mode),
            balance_each_usage(self.df[self.df['usage'] == 'test'], mode)
        ],
                         ignore_index=True)

    def get_df(self,
               mode='ANN',
               with_img = False,
               drawlandmarks=False,
               rotate=True,
               repos=True,
               mapping=False,
               cmap='GRAY',
               sample=False,
               sample_size=10):
        if sample is False:
            sample_size = self.len
        print('Generate df with config')
        print(
            f' mode:{mode}\n drawlandmarks:{drawlandmarks}\n mapping:{mapping}\n colormap:{cmap}\n sample:{sample}\n sample_size:{sample_size}'
        )
        if sample == True:
            self.df = pd.concat([
                self.df[self.df[' Usage'] == 'Training'].sample(n=sample_size),
                self.df[self.df[' Usage'] == 'PublicTest'].sample(
                    n=sample_size),
                self.df[self.df[' Usage'] == 'PrivateTest'].sample(
                    n=sample_size)
            ])
            print('Prepare data to img')
            self.df['img'] = self.df.progress_apply(to_img, axis=1)
            self.df = getlandmark(self.df, mode, drawlandmarks, mapping, cmap, with_img, rotate, repos)
            return self.df
        else:
            print('Prepare data to img')
            self.df['img'] = self.df.progress_apply(to_img, axis=1)
            self.df = getlandmark(self.df, mode, drawlandmarks, mapping, cmap, with_img, rotate, repos)
            return self.df

def re_pos(mesh):
    # print(f'shift from: {mesh[1]}')
    origin = mesh[1] - np.array([0, 0, 0])
    mesh = mesh - origin
    return mesh

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def rot(mesh, u,v,w, x,y,z):
    r = R.from_matrix([u,v,w])
    ref_r = R.from_matrix([x,y,z])
    rotation = ref_r * r
    # mesh = r.apply(mesh)
    # mesh = ref_r.apply(mesh)
    mesh = rotation.apply(mesh)
    # print(r.as_euler('xyz', degrees=True))
    return mesh

def get_ref(mesh):
    p0 = mesh[1]
    p1 = mesh[5]
    p2 = mesh[44]
    p3 = mesh[274]
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p0 - p1
    v4 = p0 - mesh[4]
    w = unit_vector(np.cross(v1, v2))
    ref = unit_vector(np.cross(v3, v4))
    v = unit_vector(np.cross(w, ref))
    u = unit_vector(np.cross(v, w))
    return u,v,w

def balance_each_usage(df, mode):
    print(f'Balancing dataFrame with {mode}sampling mode')
    target_list = list(df['target'].value_counts().index)
    if mode == 'up':
        count = df['target'].value_counts().max()
    elif mode == 'down':
        count = df['target'].value_counts().min()
    list_df = [
        df[df['target'] == t].sample(n=count,
                                     random_state=1,
                                     replace=(mode == 'up'))
        for t in target_list
    ]
    print(f'target list:{target_list}')
    print(f'count:{count}')
    return pd.concat(list_df, ignore_index=True).sample(frac=1,
                                                        ignore_index=True)


def to_img(row):
    return np.array(row[' pixels'].split(' ')).reshape(48, 48).astype('uint8')


def prepareforANN(lanmarks, rotation=True, reposition=True):
    mesh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in lanmarks])
    if reposition == True:
        mesh = re_pos(mesh)
    if rotation == True:
        u,v,w = get_ref(mesh)
        x = np.array([1., 0., 0.])
        y = np.array([0., -1., 0.])
        z = np.array([0., 0., -1.])
        mesh = rot(mesh, u,v,w, x,y,z)
    return mesh

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
    return FERclassName.index(emotion)

def getlandmark(df, mode, draw, map, cmap, with_img, rotate, repos):
    new_df = pd.DataFrame(columns=['usage', 'feature', 'target'])
    draw_img, real_img = [], []
    usage_map = {
        'Training': 'train',
        'PublicTest': 'val',
        'PrivateTest': 'test',
    }
    print('Getting landmarks with mediapipe FaceMesh')
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

            if mode == 'ANN' or mode == 'GNN':
                result = face_mesh.process(img)
                if not result.multi_face_landmarks:
                    continue

                np_landmark = prepareforANN(
                    result.multi_face_landmarks[0].landmark,rotation=rotate,reposition=repos)

                if draw == True:
                    annotated_image = cv2.cvtColor(data[1],
                                                   cv2.COLOR_GRAY2RGB).copy()
                    annotated_image = drawalllandmark(annotated_image, result)
                    draw_img.append(annotated_image)
                
                if with_img == True:
                    real_img.append(img)
                    
                new_df = pd.concat([
                    new_df,
                    pd.DataFrame([[usage_map[data[0]], np_landmark, emotion]],
                                 columns=['usage', 'feature', 'target'])
                ],
                                   ignore_index=True)
                if mode == 'GNN':
                    edge_index = list(mp_face_mesh.FACEMESH_TESSELATION)
                    new_df['edge_index'] = [edge_index for _ in range(len(new_df))]

            elif mode == 'IMG':
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
                print('Currently have only ANN, IMG, and GNN mode')
                break

        if draw is True:
            new_df['draw_img'] = draw_img
        
        if with_img is True:
            new_df['img'] = real_img

        print(
            f'Distribution of Train: \n{new_df[new_df["usage"] == "train"]["target"].value_counts()}'
        )
        print(
            f'Distribution of Validation: \n{new_df[new_df["usage"] == "val"]["target"].value_counts()}'
        )
        print(
            f'Distribution of Test \n{new_df[new_df["usage"] == "test"]["target"].value_counts()}'
        )

    return new_df