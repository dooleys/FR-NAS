# python3 face.evoLVe.PyTorch/applications/align/face_align_pre-landmarked.py --source_root data/CelebA/Img/from_zip --dest_root data/CelebA/Img/img_align_celeba_landmark_cropped --landmark_file data/CelebA/Anno/list_landmarks_align_celeba.txt
from PIL import Image
from detector import detect_faces
from align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm
import argparse
import pandas as pd

def get_landmarks(df, image_id):
    '''
    Given a landmark dataframe and an image id, return an list of 5 tuples
    '''
    row = df.loc[image_id]
    return [(row['lefteye_x'],row['lefteye_y']),
            (row['righteye_x'],row['righteye_y']),
            (row['nose_x'],row['nose_y']),
            (row['leftmouth_x'],row['leftmouth_y']),
            (row['rightmouth_x'],row['rightmouth_y'])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "face alignment")
    parser.add_argument("-source_root", "--source_root", help = "specify your source dir", default = "./data/test", type = str)
    parser.add_argument("-dest_root", "--dest_root", help = "specify your destination dir", default = "./data/test_Aligned", type = str)
    parser.add_argument("-landmark_file", "--landmark_file", help = "specify your destination dir", default = "./data/landmark_file.txt", type = str)
    parser.add_argument("-crop_size", "--crop_size", help = "specify size of aligned faces, align and crop with padding", default = 112, type = int)
    args = parser.parse_args()

    source_root = args.source_root # specify your source dir
    dest_root = args.dest_root # specify your destination dir
    landmarks = args.landmark_file # specify the landmarks 
    crop_size = args.crop_size # specify size of aligned faces, align and crop with padding
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square = True) * scale

    cwd = os.getcwd() # delete '.DS_Store' existed in the source_root
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)
    
    if landmarks:
        # resulting dataframe has image names as the index 
        # and ten cloumns for left and right eye, nose, and left and right mouth
        df = pd.read_csv(landmarks, delim_whitespace=True, header=1)

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)

    for subfolder in tqdm(os.listdir(source_root)):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
            img = Image.open(os.path.join(source_root, subfolder, image_name))
            try: # Handle exception
                landmarks = get_landmarks(df, os.path.basename(image_name))
            except Exception:
                print("{} is discarded due to exception!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
                print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, subfolder, image_name)))
                continue
#             facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            facial5points = landmarks
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']: #not from jpg
                image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
            img_warped.save(os.path.join(dest_root, subfolder, image_name))
