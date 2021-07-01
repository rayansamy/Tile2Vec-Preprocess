import time, os
from math import cos, sin
import random
import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

TILE_FILENAME_FORMAT = "{triplet_id:08d}_{tile_type}.png"

def transform(input_path, output_path, size, plot=False):
    samples = []
    to_be_saved = {}
    for path, subdirs, files in os.walk(input_path):
        for name in files:
            samples.append(os.path.join(path, name))

    train_saving_path = os.path.join(output_path, 'train')
    study_saving_path = os.path.join(output_path, 'study')
    saving_path = train_saving_path

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(train_saving_path):
        os.mkdir(train_saving_path)
    if not os.path.isdir(study_saving_path):
        os.mkdir(study_saving_path)
    
    PI = 3.1456
    t = 0
    index_img = 0
    random.shuffle(samples)
    for sample in tqdm.tqdm(samples):
        if index_img < 0.81*len(samples):
            saving_path = train_saving_path
        else:
            saving_path = study_saving_path
        img_path = sample
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if plot:
            try:
                plt.figure()
                plt.imshow(img)
                plt.suptitle("Anchor img")
            except:
                print("passed img")
        #print("Img shape : "+str(img.shape))
        for x in range(0, img.shape[0], 256):
            for y in range(0, img.shape[1], 256):
                if img[x:x+256,y:y+256].shape != (256, 256, 3) or  [0,0,0] in img[x:x+256,y:y+256]:
                    continue
                start = time.time()
                #time.clock()    
                found = False # Variable meaning that we didn't find suiting neighobr tile
                best_neighbor = 0
                best_neighbor_img = []
                while not found: # Getting a neighbor image
                    R = 256/2
                    theta = random.random() * 2 * PI

                    
                    x_shift, y_shift = int(R * cos(theta)), int(R * sin(theta))
                    neighobr_img = img[x+x_shift:x+x_shift+256,y+y_shift:y+y_shift+256]
                    #print("Neighbor Img shape : "+str(neighobr_img.shape))
                    if plot:
                        try:
                            plt.figure()
                            plt.imshow(neighobr_img)
                            plt.suptitle("Neighbor img")
                        except:
                            print("passed img")
                    elapsed = time.time() - start
                    #print("Elapsed : "+str(elapsed))
                    if len(np.unique(neighobr_img.reshape(-1, neighobr_img.shape[2]), axis=0))>best_neighbor and neighobr_img.shape == (256, 256, 3):
                        best_neighbor = len(np.unique(neighobr_img.reshape(-1, neighobr_img.shape[2]), axis=0))
                        best_neighbor_img = neighobr_img

                    if (not (neighobr_img.shape != (256, 256, 3) or len(np.unique(neighobr_img.reshape(-1, neighobr_img.shape[2]), axis=0))<30)) or elapsed>10:
                        found = True
                """
                if [0, 0, 0] in best_neighbor_img:
                    continue
                """
                neighobr_img = best_neighbor_img
                found = False # distant
                while not found:
                    #print("\n\nCheckpoint 3\n\n")
                    random_img = samples[random.randint(0, len(samples)-1)]
                    if random_img == sample:
                        continue
                    distant_img_path = random_img
                    distant_img = cv2.imread(distant_img_path, 1)
                    distant_img = cv2.cvtColor(distant_img, cv2.COLOR_BGR2RGB)

                    if plot:
                        try:
                            plt.figure()
                            plt.imshow(distant_img)
                            plt.suptitle("Distant img")
                        except:
                            pass
                    #print("Distant Img shape : "+str(distant_img.shape))
                    #distant_img = cv2.cvtColor(distant_img, cv2.COLOR_BGR2RGB)
                    x_distant = random.randint(0, distant_img.shape[0]-257)
                    y_distant = random.randint(0, distant_img.shape[1]-257)
                    distant_img_shaped = distant_img[x_distant:x_distant+256,y_distant:y_distant+256]
                    if not (distant_img_shaped.shape != (256, 256, 3) or [0,0,0] in distant_img_shaped):
                        found = True
                    
                tm = t
                # anchor
                out_name = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='anchor')
                out_name = os.path.join(saving_path, out_name)
                past_img = img[x:x+256,y:y+256]
                past_img = cv2.cvtColor(past_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(out_name, past_img)

                # neighbor
                out_name_neighbor = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='neighbor')
                out_name_neighbor = os.path.join(saving_path, out_name_neighbor)
                neighobr_img = cv2.cvtColor(neighobr_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(out_name_neighbor, neighobr_img)

                # distant
                out_name_distant = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='distant')
                out_name_distant = os.path.join(saving_path, out_name_distant)
                distant_img_shaped = cv2.cvtColor(distant_img_shaped, cv2.COLOR_BGR2RGB)
                cv2.imwrite(out_name_distant, distant_img_shaped)
                to_be_saved[str(tm)] = sample
                t += 1
        index_img += 1
    
    with open(os.path.join( output_path, 'indexes.json'), 'w') as fp:
        json.dump(to_be_saved, fp)
    print("Everything's done ! :)")