import os
import requests
import zipfile


def main():
    glove_url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    download_path = 'glove.6B.zip'
    extract_path = './glove.6B'
    
    if not os.path.exists(download_path):
        print(f'Downloading GloVe embeddings from {glove_url}...')
        resp = requests.get(glove_url, stream=True)
        with open(download_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=128):
                f.write(chunk)
        print('GloVe embeddings downloaded.')
    else:
        print('GloVe embeddings zip file found.')

    if not os.path.exists(extract_path):
        print(f'Extracting GloVe embeddings...')
        with zipfile.ZipFile(download_path, 'r') as f:
            f.extractall(extract_path)
        print('GloVe embeddings extracted.')
    else:
        print('GloVe embeddings extraction found.')


if __name__ == '__main__':
    main()