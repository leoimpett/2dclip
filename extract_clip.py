

# options
# image_dir = "/Users/impett/Documents/Code/Datasets/fitz_urls_only/fitz_images_red.csv"
image_dir = "/Users/impett/Documents/Code/Datasets/brasil-candido-portinari/merged"
# image_dir = "/Users/impett/Documents/Code/Datasets/dset_COCO"
save_compressed = False
# if save_compressed, will both round numbers to 3 decimal places, and save as a gzip file




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


import requests
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_clip_vectors(image_path):
    # if image_path is a url, download it
    if image_path[:4] == "http":
        response = requests.get(image_path)
        PIL_image = Image.open(BytesIO(response.content))
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

    with open_func(output_file, write_option) as f:
        for j, image_path in enumerate(tqdm.tqdm(image_paths)):
            image_features = get_clip_vectors(image_path)
            if save_compressed:
                vec_string = "["+",".join([str(round(x,3)) for x in image_features[0]])+"]"
            else:
                vec_string = "["+",".join([str(x) for x in image_features[0]])+"]"

            if isUrl:
                image_catalog_url = image_catalog_dict[image_path]
                metadata = {"catalog": image_catalog_url}
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


extract_clip_vectors(image_dir)

