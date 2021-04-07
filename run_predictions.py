import os
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
from red_light_detector import RedLightDetector

# set the path to the downloaded data:
data_path = '../data/redlightdata'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds'

os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in tqdm(range(len(file_names))):
    rld = RedLightDetector()

    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[i]))

    # convert to numpy array:
    I = np.asarray(I).astype(float)

    preds[file_names[i]] = rld.detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
