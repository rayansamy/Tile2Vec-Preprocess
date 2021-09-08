import time, os
from math import cos, sin
import random
from tqdm.notebook import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pathlib
import pickle
import pandas as pd
import operator


TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.png"

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
    for sample in tqdm(samples):
        if index_img < 0.80*len(samples):
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


def transform_multi_scale(input_path, output_path, pd_file, scales):
    divisions = scales
    df = pd.read_csv(pd_file)
    df['Image'] = df['Image_Label'].map(lambda x: x.split('_')[0])
    df['Class'] = df['Image_Label'].map(lambda x: x.split('_')[1])
    classes = df['Class'].unique()
    train_df = df.groupby('Image')['Class'].agg(set).reset_index()
    for class_name in classes:
        df[class_name] = df['Class'].map(lambda x: 1 if class_name in x else 0)
    df.head()

    image_col = np.array(df['Image'])
    image_files = image_col[::4]
    all_labels = df['Class']
    X_train, y_train = image_files, all_labels
    train_pairs = np.array(list(zip(X_train, y_train)))
    TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.png"
    NUM_TRAIN_SAMPLES = len(train_pairs)
    train_samples = train_pairs[np.random.choice(train_pairs.shape[0], NUM_TRAIN_SAMPLES, replace=False), :]


    shown = False
    for div in reversed(divisions):
        print("DIV : "+str(div))
        labs = []
        shown = True
        t = 0
        associations = ['Fish', 'Flower', 'Gravel', 'Sugar']
        PATH_TRAIN = output_path+"MULTI_SCALE/"+str(div)+"/train/"
        if not Path(PATH_TRAIN).is_dir():
            p = pathlib.Path(PATH_TRAIN)
            p.mkdir(parents=True, exist_ok=True)
        print("train samples : "+str(train_samples))
        for sample in tqdm(train_samples):
                img_path = os.path.join(input_path, 'train_images', sample[0])
                img = cv2.imread(img_path, 1)
                labels = df[df['Image_Label'].str.contains(sample[0])]['EncodedPixels']
                for idx, rle in enumerate(labels.values):
                    if rle is not np.nan:
                        mask = rle2mask(rle)
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        rgbImg = extract_contour(img, contours, shown=shown, direct=True)

                        for x in range(0, rgbImg.shape[0], div):
                            for y in range(0, rgbImg.shape[1], div):
                                if rgbImg[x:x+div,y:y+div].shape != (div, div, 3) or [0,0,0] in rgbImg[x:x+div,y:y+div]:
                                    continue
                                tm = t
                                if t == 10000:
                                    break
                                grgr = cv2.resize(rgbImg[x:x+div,y:y+div], (256, 256), interpolation = cv2.INTER_AREA)
                                out_name = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='anchor')
                                
                                out_name = PATH_TRAIN+out_name
                                cv2.imwrite(out_name, grgr)
                                t += 1
                                labs.append(idx)
                        
                if t == 10000:
                    break
        with open(output_path+"MULTI_SCALE/"+str(div)+"/labels.pickle", 'wb') as handle:
            pickle.dump(labs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def transform_biggest_square(input_path, output_path, pd_file): 
    df = pd.read_csv(pd_file)
    df['Image'] = df['Image_Label'].map(lambda x: x.split('_')[0])
    df['Class'] = df['Image_Label'].map(lambda x: x.split('_')[1])
    classes = df['Class'].unique()
    train_df = df.groupby('Image')['Class'].agg(set).reset_index()
    for class_name in classes:
        df[class_name] = df['Class'].map(lambda x: 1 if class_name in x else 0)
    df.head()

    image_col = np.array(df['Image'])
    image_files = image_col[::4]
    all_labels = df['Class']
    X_train, y_train = image_files, all_labels
    train_pairs = np.array(list(zip(X_train, y_train)))
    TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.png"
    NUM_TRAIN_SAMPLES = len(train_pairs)
    train_samples = train_pairs[np.random.choice(train_pairs.shape[0], NUM_TRAIN_SAMPLES, replace=False), :]
    
    labs = []
    shown = False
    t = 0
    patches = []
    associations = ['Fish', 'Flower', 'Gravel', 'Sugar','Undefined']
    PATH_TRAIN = output_path + "CLUSTERING_BIGGEST_SQUARE/train/"
    pathlib.Path(PATH_TRAIN).mkdir(parents=True, exist_ok=True)
    a = 0
    print("LEN : "+str(len(train_samples)))
    for sample in tqdm(train_samples):
            img_path = input_path+'train_images/'+str(sample[0])
            #print("img path : "+str(img_path))
            img = cv2.imread(img_path, 1)
            #print(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            labels = df[df['Image_Label'].str.contains(sample[0])]['EncodedPixels']
            patches = []
            for idx, rle in enumerate(labels.values):
                if rle is not np.nan:
                    mask = rle2mask(rle)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    rgbImg = extract_contour(img, contours, shown=shown, direct=True)
                    min_shape = min(rgbImg.shape[0], rgbImg.shape[1])
                    
                    
                    rgbImg = cv2.resize(rgbImg[0:min_shape, 0:min_shape], (256, 256), interpolation = cv2.INTER_AREA)
                    if [0,0,0] not in rgbImg and min_shape>=128:
                        labs.append(idx)
                                    
                        tm = t
                        out_name = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='anchor')
                        
                        out_name = PATH_TRAIN+out_name
                        #print("out name : "+str(out_name))
                        cv2.imwrite(out_name, rgbImg)
                        t += 1
                
                    for c in contours:
                        img = cv2.drawContours(img, [c], -1, (0,0,0), -1)

            undef = extract_non_black(img)
            undef = cv2.resize(undef, (256, 256), interpolation = cv2.INTER_AREA)
            
            if [0,0,0] not in undef and undef.shape[0]>=128:
                tm = t
                out_name = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='anchor')
                """
                fig = plt.figure()
                plt.imshow(undef)
                fig.suptitle(out_name, fontsize=20)
                plt.show()
                """
                labs.append(4)
                out_name = PATH_TRAIN+out_name
                cv2.imwrite(out_name, undef)
                t += 1

    with open(output_path+"/CLUSTERING_BIGGEST_SQUARE/labels.pickle", 'wb') as handle:
        pickle.dump(labs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def transform_biggest_square_classification(input_path, output_path, pd_file):
    df = pd.read_csv(pd_file)
    df['Image'] = df['Image_Label'].map(lambda x: x.split('_')[0])
    df['Class'] = df['Image_Label'].map(lambda x: x.split('_')[1])
    classes = df['Class'].unique()
    train_df = df.groupby('Image')['Class'].agg(set).reset_index()
    for class_name in classes:
        df[class_name] = df['Class'].map(lambda x: 1 if class_name in x else 0)
    df.head()

    image_col = np.array(df['Image'])
    image_files = image_col[::4]
    all_labels = df['Class']
    X_train, y_train = image_files, all_labels
    train_pairs = np.array(list(zip(X_train, y_train)))
    TILE_FILENAME_FORMAT = "{triplet_id:05d}_{tile_type}.png"
    NUM_TRAIN_SAMPLES = len(train_pairs)
    train_samples = train_pairs[np.random.choice(train_pairs.shape[0], NUM_TRAIN_SAMPLES, replace=False), :]
    

    labs = []
    shown = True
    t = 0
    associations = ['Fish', 'Flower', 'Gravel', 'Sugar']
    PATH_TRAIN = output_path +"CLASSIF_ALL_IMAGE/"
    pathlib.Path(PATH_TRAIN).mkdir(parents=True, exist_ok=True)
    count_class = {}; count_class["Fish"] = 0; count_class["Flower"] = 0; count_class["Gravel"] = 0; count_class["Sugar"] = 0;
    for classe in count_class.keys():
        pathlib.Path(PATH_TRAIN+classe).mkdir(parents=True, exist_ok=True)
    stats = {}
    for sample in tqdm(train_samples):
            if not shown:
                fig, ax = plt.subplots(figsize=(15, 10))
            img_path = os.path.join(input_path, 'train_images', sample[0])
            img = cv2.imread(img_path, 1)
            labels = df[df['Image_Label'].str.contains(sample[0])]['EncodedPixels']
            patches = []
            dominance = 0
            recapitulatif = {}
            all_size = (0, 0, 0)
            for idx, rle in enumerate(labels.values):
                if rle is not np.nan:
                    mask = rle2mask(rle)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    rgbImages = extract_contour(img, contours, shown=shown, direct=False)
                    summed = (0, 0)
                    for rgbImg in rgbImages:
                        summed = tuple(map(operator.add, summed, rgbImg.shape))
                    all_size = (all_size[0]+summed[0], all_size[1]+summed[1])
                    recapitulatif[str(idx)] = (summed[0],summed[1])
                                
                recapitulatif_percentage = {}
            for key in recapitulatif:
                recapitulatif_percentage[key] = (recapitulatif[key][0]/all_size[0] + recapitulatif[key][1]/all_size[1])/2
            best = 0
            best_idx = -1
            for key in recapitulatif_percentage:
                if recapitulatif_percentage[key]>best:
                    best = recapitulatif_percentage[key]
                    best_idx = key
            if best>0.64:
                out_name = PATH_TRAIN+""+associations[int(best_idx)]+"/"+str(count_class[str(associations[int(best_idx)])])+".jpg"

                print("out_name "+str(out_name))
                cv2.imwrite(out_name, img)
                count_class[str(associations[int(best_idx)])] += 1
                


def rle2mask(mask_rle, shape=(2100, 1400)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def extract_non_black(img):
    # Import your picture
    input_picture = img.copy()
    #input_picture = cv2.cvtColor(input_picture, cv2.COLOR_BGR2RGB) 
    # Color it in gray
    gray = cv2.cvtColor(input_picture, cv2.COLOR_BGR2GRAY)

    # Create our mask by selecting the non-zero values of the picture
    ret, mask = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY)

    # Select the contour
    cont, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if your mask is incurved or if you want better results, 
    # you may want to use cv2.CHAIN_APPROX_NONE instead of cv2.CHAIN_APPROX_SIMPLE, 
    # but the rectangle search will be longer
    
    cv2.drawContours(input_picture, cont, -1, (0,255,255), 2)
    """
    plt.figure()
    plt.imshow(input_picture)
    plt.show()
    """
    # Find contour and sort by contour area
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        #return ROI
        break
    mina = min(ROI.shape[0], ROI.shape[1])
    return ROI[0:mina, 0:mina]
    plt.figure()
    plt.imshow(ROI[0:mina, 0:mina])
    plt.show()
    

def extract_contour(img, cnts, shown=False, direct=True):
    '''
        Extract from contours images
    '''
    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rois = []
    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        if direct:
            return ROI
        rois.append(ROI)
    if not shown:
        plt.figure()
        plt.imshow(ROI)
    return rois



def predict_domain(image):
    '''
        for a specific image of zooniverse predict each tile.
    '''
    deb_path = "../../tmp/train/"
    i = 0
    for x in range(0, image.shape[0], 256):
            for y in range(0, image.shape[1], 256):
                out_name =  TILE_FILENAME_FORMAT.format(triplet_id=i,tile_type='anchor')
                out_name = deb_path+out_name
                print("out_name : "+str(out_name)+" shape : "+str(image[x:x+256,y:y+256].shape))
                
                if image[x:x+256,y:y+256].shape != (256, 256, 3):
                    continue
                to_save = cv2.cvtColor(image[x:x+256,y:y+256], cv2.COLOR_BGR2RGB)
                cv2.imwrite(out_name, to_save)
                i += 1
    print("Number of images : "+str(i))
    dataset_path =  'tmp/'
    tile_dataset = ImageSingletDataset(data_dir=dataset_path, tile_type=TileType.ANCHOR)
    da_embeddings = get_embeddings(tile_dataset=tile_dataset, model=model)
    compressed = pca.fit_transform(da_embeddings)
    print("Embedding shape : "+str(da_embeddings.shape))
    print("Compressed PCA shape : "+str(compressed.shape))
    plt.imshow(compressed.T)
    return da_embeddings
    

def visualize_boxes_cluster(df, sample):
    tmp_dataset_path =  '../../../NC/tmp'
    for filename in os.listdir(tmp_dataset_path+"/train/"):
        file_path = os.path.join(tmp_dataset_path+"/train/", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    coordinates = []
    t = 0
    
    fig, ax = plt.subplots(figsize=(15, 10))
    img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get annotations
    labels = df[df['Image_Label'].str.contains(sample[0])]['EncodedPixels']
    patches = []
    for idx, rle in enumerate(labels.values):
        if rle is not np.nan:
            mask = rle2mask(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cnts = sorted(contours, key=cv2.contourArea, reverse=True)
            rois = []
            # Find bounding box and extract ROI
            xvs,yvs,wvs,hvs = [], [], [], []
            for c in cnts:
                xv,yv,wv,hv = cv2.boundingRect(c)
                xvs.append(xv)
                yvs.append(yv)
                wvs.append(wv)
                hvs.append(hv)
                #break
            #print("Contours : "+str(contours))
            rgbImages = extract_contour(img, contours, direct=False)
            da = 0
            for rgbImg in rgbImages:
                rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
                for x in range(0, rgbImg.shape[0], 256):
                    for y in range(0, rgbImg.shape[1], 256):
                        if rgbImg[x:x+256,y:y+256].shape != (256, 256, 3):# or  [0,0,0] in rgbImg[x:x+256,y:y+256]:
                            continue
                        
                        tm = t
                        coordinates.append([xvs[da]+x,yvs[da]+y, wvs[da], hvs[da]])
                        out_name = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='anchor')
                        out_name = '../../../NC/tmp/train/'+out_name
                        cv2.imwrite(out_name,rgbImg[x:x+256,y:y+256])
                        t += 1
                da += 1
            for contour in contours:
                poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=2, edgecolor=COLORS[idx], facecolor=COLORS[idx], fill=True)
                patches.append(poly_patch)
    p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet, alpha=0.3)
    ax.imshow(img/255)
    ax.set_title('{} - ({})'.format(sample[0], ', '.join(sample[1].astype(np.str))))
    ax.add_collection(p)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
    
    tmp_tile_dataset = ImageSingletDataset(data_dir=tmp_dataset_path, tile_type=TileType.ANCHOR)
    tmp_da_embeddings = get_embeddings(tile_dataset=tmp_tile_dataset, model=model)
    tmp_clusters = convml_tt.interpretation.plots.dendrogram(tmp_da_embeddings, n_samples=10, n_clusters_max=12, return_clusters=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['red', 'azure', 'forestgreen', 'mediumblue', 'goldenrod', 'olivedrab', 'chocolate', 'darkseagreen', 'steelblue', 'dodgerblue', 'crimson', 'darkorange']
    # Display the image
    #rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
    ax.imshow(img/255)

    # Create a Rectangle patch
    cnt = 0
    for coordinate in coordinates:
        rect = rectan((coordinate[0], coordinate[1]), coordinate[2], coordinate[3], linewidth=1, facecolor=colors[tmp_clusters[1][cnt]]  , edgecolor=colors[tmp_clusters[1][cnt]], fill=True,  alpha=0.3)
        #poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=2, edgecolor=COLORS[idx], facecolor=COLORS[idx], fill=True)
        #patches.append(poly_patch)
        cnt += 1
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.set_title("Per tile cluster prediction")
    ax.set_ylabel("KM")
    ax.set_xlabel("KM")
    plt.show()


def visualize_domain(df, sample):
    tmp_dataset_path =  '../../../NC/tmp'
    for filename in os.listdir(tmp_dataset_path+"/train/"):
        file_path = os.path.join(tmp_dataset_path+"/train/", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    coordinates = []
    t = 0
    
    fig, ax = plt.subplots(figsize=(15, 10))
    img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get annotations
    labels = df[df['Image_Label'].str.contains(sample[0])]['EncodedPixels']

    patches = []
    
    for x in range(0, img.shape[0], 256):
        for y in range(0, img.shape[1], 256):
            if img[x:x+256,y:y+256].shape != (256, 256, 3):# or  [0,0,0] in rgbImg[x:x+256,y:y+256]:
                continue
            tm = t
            coordinates.append([x, y, 256, 256])
            out_name = TILE_FILENAME_FORMAT.format(triplet_id=tm,tile_type='anchor')
            out_name = '../../../NC/tmp/train/'+out_name
            cv2.imwrite(out_name, img[x:x+256,y:y+256])
            t += 1
    for idx, rle in enumerate(labels.values):
        if rle is not np.nan:
            mask = rle2mask(rle)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=2, edgecolor=COLORS[idx], facecolor=COLORS[idx], fill=True)
                patches.append(poly_patch)
    p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet, alpha=0.3)
    ax.imshow(img/255)
    ax.set_title('{} - ({})'.format(sample[0], ', '.join(sample[1].astype(np.str))))
    ax.add_collection(p)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
    
    tmp_tile_dataset = ImageSingletDataset(data_dir=tmp_dataset_path, tile_type=TileType.ANCHOR)
    tmp_da_embeddings = get_embeddings(tile_dataset=tmp_tile_dataset, model=model)
    tmp_clusters = convml_tt.interpretation.plots.dendrogram(tmp_da_embeddings, n_samples=10, n_clusters_max=12, return_clusters=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['red', 'azure', 'forestgreen', 'mediumblue', 'goldenrod', 'olivedrab', 'chocolate', 'darkseagreen', 'steelblue', 'dodgerblue', 'crimson', 'darkorange']
    # Display the image
    ax.imshow(img/255)

    # Create a Rectangle patch
    cnt = 0
    for coordinate in coordinates:
        rect = rectan((coordinate[1], coordinate[0]), coordinate[2], coordinate[3], linewidth=1, facecolor=colors[tmp_clusters[1][cnt]]  , edgecolor=colors[tmp_clusters[1][cnt]], fill=True,  alpha=0.3)
        cnt += 1
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.set_title("Per tile cluster prediction")
    ax.set_ylabel("KM")
    ax.set_xlabel("KM")
    plt.show()






def kmeans(
    da_embeddings,
    ax=None,
    visualize=False,
    model_path=None,
    n = 12,
    method=None,
    save=False
):
    """
    K-Means clustering.
    """

    tile_dataset = ImageSingletDataset(
        data_dir=da_embeddings.data_dir,
        tile_type=da_embeddings.tile_type,
        stage=da_embeddings.stage,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))
    else:
        fig = ax.figure

    if model_path is None:
        if method == "optimize":
            sse = []
            silhouette_coefficients = []
            best_silouhette = -999
            best_silouhette_k = -1
            best_clusters = []
            r = 0
            for k in range(2, 30):
                kmeans = KMeans(n_clusters=k)
                clusters = kmeans.fit_predict(da_embeddings)
                sse.append(kmeans.inertia_)
                #centers = clusterer.cluster_centers_
                score = silhouette_score(da_embeddings, clusters)
                silhouette_coefficients.append(score)
                if score>best_silouhette:
                    best_silouhette = score
                    best_silouhette_k = kmeans
                    best_clusters = clusters
            kmeans = best_silouhette_k
            clusters = best_clusters
            if visualize:
                fig = plt.gcf()
                fig.set_size_inches(20, 20)
                plt.style.use("fivethirtyeight")
                plt.plot(range(2, 30), sse)
                plt.xticks(range(1, 30))
                plt.xlabel("Number of Clusters")
                plt.ylabel("SSE")
                plt.show()

                fig = plt.gcf()
                fig.set_size_inches(20, 20)
                plt.style.use("fivethirtyeight")
                plt.plot(range(2, 30), silhouette_coefficients)
                plt.xticks(range(2, 30))
                plt.xlabel("Number of Clusters")
                plt.ylabel("Silhouette Coefficient")
                plt.show()
        else:
            kmeans = KMeans(
                    init="random",
                    n_clusters=n,
                    n_init=10,
                    max_iter=300,
                    random_state=42 
                )
            clusters = kmeans.fit_predict(da_embeddings)
    else:
        kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
        clusters = kmeans.predict(da_embeddings)
        return clusters, kmeans
    if save:
        pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))    

    return clusters, kmeans