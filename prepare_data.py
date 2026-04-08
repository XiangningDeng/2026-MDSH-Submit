import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import zipfile
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources

# Configuration Options
# PREPARATION_MODE: 'official' for pretrained GloVe, 'random' for fast init
PREPARATION_MODE = 'official' 

# Mirror URL reference:
# https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDlarge_utils.zip
MIRROR_URL = r'https://huggingface.co/datasets/Recommenders/MIND/resolve/main/'

base_path = os.getcwd()
utils_path = os.path.join(base_path, "utils")
os.makedirs(utils_path, exist_ok=True)

# 1. Build dictionaries from local data (Ensures 100 percent ID matching)
print(f"Mode: {PREPARATION_MODE} | Step 1: Building dictionaries from local files")
train_news = os.path.join(base_path, "data", "train", "news.tsv")
valid_news = os.path.join(base_path, "data", "valid", "news.tsv")
train_behaviors = os.path.join(base_path, "data", "train", "behaviors.tsv")

def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

words, verts, subverts = set(), set(), set()
for f in [train_news, valid_news]:
    if os.path.exists(f):
        df = pd.read_csv(f, sep='\t', header=None, quoting=3, names=['id', 'vert', 'subvert', 'title', 'abstract', 'url', 't_ent', 'a_ent'])
        verts.update(df['vert'].dropna().unique())
        subverts.update(df['subvert'].dropna().unique())
        for text in tqdm(df['title'].dropna(), desc=f"Reading {os.path.basename(f)}"):
            words.update(clean_text(text).split())
        for text in tqdm(df['abstract'].dropna(), desc="Processing abstracts"):
            words.update(clean_text(text).split())

word_dict = {word: i + 1 for i, word in enumerate(sorted(list(words)))}
word_dict['<pad>'] = 0
vert_dict = {v: i + 1 for i, v in enumerate(sorted(list(verts)))}
subvert_dict = {sv: i + 1 for i, sv in enumerate(sorted(list(subverts)))}

# 2. Handle word embeddings
if PREPARATION_MODE == 'official':
    print("Step 2: Downloading official Utils from Hugging Face mirror")
    try:
        # Download and extract the resources
        download_deeprec_resources(MIRROR_URL, utils_path, 'MINDlarge_utils.zip')
        print("Success: Official embeddings are ready in embedding_all.npy")
    except Exception as e:
        print(f"Error: Download failed, falling back to random mode: {e}")
        PREPARATION_MODE = 'random'

if PREPARATION_MODE == 'random':
    print("Step 2: Generating randomly initialized embeddings")
    embeddings = np.random.normal(scale=0.1, size=(len(word_dict), 300)).astype(np.float32)
    np.save(os.path.join(utils_path, "embedding_all.npy"), embeddings)
    print("Success: Random embeddings generated")

# 3. Save dictionaries
with open(os.path.join(utils_path, "word_dict_all.pkl"), "wb") as f: pickle.dump(word_dict, f)
with open(os.path.join(utils_path, "vert_dict.pkl"), "wb") as f: pickle.dump(vert_dict, f)
with open(os.path.join(utils_path, "subvert_dict.pkl"), "wb") as f: pickle.dump(subvert_dict, f)

print(f"\nRefinement completed. Vocab size: {len(word_dict)}, Categories: {len(vert_dict)}")
