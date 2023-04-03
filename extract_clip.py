

# options
image_dir = "/Users/impett/Documents/Code/Datasets/fitz_urls_only/fitz_images_red.csv"
# image_dir = "/Users/impett/Documents/Code/Datasets/brasil-candido-portinari/merged"
# image_dir = "/Users/impett/Documents/Code/Datasets/dset_COCO"
save_compressed = False
# if save_compressed, will both round numbers to 3 decimal places, and save as a gzip file


write_json = True

# Expects 2 columns in the CSV - "thumb" and "catalog". The embeddings will be taken from the thumb, which will also be
# passed as the texture basis. Links will instead go to "catalog".... 


max_images = 2000 


base_dir = "/".join(image_dir.split("/")[:-1])

# strip last / if present of image_dir
if image_dir[-1] == "/":
    image_dir = image_dir[:-1]


# this function is an alternative to running the clip extraction in the browser, intended for larger datasets

import torch
import clip
from PIL import Image

import glob
import tqdm
import gzip
import numpy as np
import os

import requests
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_clip_vectors(image_path):
    # if image_path is a url, download it
    if image_path[:4] == "http":
        image_local_path = base_dir+'/'+image_path.split("/")[-1]
        # check if file exists locally
        if not os.path.exists(image_local_path):
            response = requests.get(image_path)
            PIL_image = Image.open(BytesIO(response.content))
            # save it in the current directory - will make things quicker in the future
            PIL_image.save(image_local_path)
        # else open the local file
        else:
            PIL_image = Image.open(image_local_path)

    else:
        PIL_image = Image.open(image_path)
    # make sure it's in RGB
    PIL_image = PIL_image.convert("RGB")
    image = preprocess(PIL_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        return image_features.cpu().numpy() 
    

# file output of the format: 
#  ${filePath}\t${vecString}\n

def is_image_file(file_path):
    image_file_formats = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"]
    return (file_path[-4:].lower() in image_file_formats) or (file_path[-5:].lower() in image_file_formats)



def extract_clip_vectors(image_dir):

    isUrl = False
    # if image_dir is a csv
    if image_dir[-4:] == ".csv":
        import pandas as pd
        df = pd.read_csv(image_dir)
        image_paths = df["thumb"].tolist()
        image_catalog = df["catalog"].tolist()
        image_large = df["large"].tolist()
        image_iiif = df["iiif"].tolist()
        # make dict 
        image_catalog_dict = dict(zip(image_paths,image_catalog))

        isUrl = True
        output_file = "/" + "/".join(image_dir.split('/')[:-1]) + "/clip_vit_32_embeddings.tsv"
    else:
        image_file_formats = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"]
        image_paths = glob.glob(image_dir + "/*")
        # now filter to only include image file formats
        image_paths = [x for x in image_paths if is_image_file(x)]
        output_file = image_dir + "/clip_vit_32_embeddings.tsv"

    print("Found {} images".format(len(image_paths)))

    if save_compressed:
        open_func = gzip.open
        output_file += ".gz"
        write_option = "wt"
    else:
        open_func = open
        write_option = "w"



    if write_json:
        output_file = output_file.replace(".tsv", ".json")
        import json


    with open_func(output_file, write_option) as f:

        # if image_paths is longer than max_images, trim it
        if len(image_paths) > max_images:
            image_paths = image_paths[:max_images]
            # randomly select max_images 
            # import random
            # image_paths = random.sample(image_paths, max_images)

        
        if write_json:
            data = {}
            for j, image_path in enumerate(tqdm.tqdm(image_paths)):
                data[image_path] = {} 
                embeddings = get_clip_vectors(image_path)
                embeddings = np.round(embeddings, decimals=3)  # Round to 3 decimal places
                embeddings = embeddings.flatten().tolist() 
                data[image_path]['embeddings'] = embeddings
                if isUrl:
                    data[image_path]['metadata'] = {"catalog": image_catalog[j], "large": image_large[j], "iiif": image_iiif[j]}
                
            json.dump(data, f, ensure_ascii=False)
            

        else:
            for j, image_path in enumerate(tqdm.tqdm(image_paths)):
                image_features = get_clip_vectors(image_path)
                if save_compressed:
                    vec_string = "["+",".join([str(round(x,3)) for x in image_features[0]])+"]"
                else:
                    vec_string = "["+",".join([str(x) for x in image_features[0]])+"]"

    #            Only write "metadata" (i.e. the catalog link) if it's a url (i.e. not a local file
                if isUrl:
                    image_catalog_url = image_catalog_dict[image_path]
                    metadata = {"catalog": image_catalog_url, "large": image_large[j], "iiif": image_iiif[j]}
                    # if not last line
                    if j < len(image_paths)-1:
                        f.write(f"{image_path}\t{vec_string}\t{metadata}\n")
                    else:
                        f.write(f"{image_path}\t{vec_string}\t{metadata}")

                else:
                    image_path_stripped = "/" + image_path.split("/")[-1]
                    # if not last line
                    if j < len(image_paths)-1:
                        f.write(f"{image_path_stripped}\t{vec_string}\n")
                    else:
                        f.write(f"{image_path_stripped}\t{vec_string}")





# extract_clip_vectors(image_dir)

def compress_json(json_file):
    import json
    import gzip
    import shutil

    with open(json_file, 'rb') as f_in:
        with gzip.open(json_file+'.gz', 'wb') as f_out:
            data = json.load(f_in)
            # make sure that data[k].embeddings are all rounded to 3 decimals
            for k in data.keys():
                data[k]['embeddings'] = np.round(data[k]['embeddings'], decimals=3).tolist()
            # save
            # convert data to bytes-like object
            json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            # save to gzip file
            f_out.write(json_data)

            # shutil.copyfileobj(f_in, f_out)

compress_json("/Users/impett/Documents/Code/Datasets/fitz_urls_only/clip_vit_32_embeddings.json")