#!/usr/bin/env python3
import subprocess
import zipfile
import tarfile
import os
import requests
import numpy as np
import tensorflow_datasets as tfds
from datasets import load_dataset,get_dataset_config_names
from huggingface_hub import snapshot_download

def download_kaggle_competition_data(competition_name, download_path='.'):
    """
    Downloads competition data from Kaggle.
    
    Parameters:
    - competition_name: str : The name of the competition on Kaggle.
    - download_path: str : The path to the directory where the data should be downloaded.
    """
    try:
        result = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', competition_name, '-p', download_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Downloaded competition data to {download_path}")
        zip_file_path = os.path.join(download_path, f"{competition_name.split('/')[-1]}.zip")
        
        # Unzip the dataset
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print(f"Unzipped dataset to {download_path}")
        
        # Remove the zip file
        os.remove(zip_file_path)
        print(f"Removed zip file {zip_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading competition data: {e.stderr.decode('utf-8')}")

def download_and_unzip_file_from_figshare(url, download_path='.'):
    """
    Downloads and unzips a file from Figshare.
    
    Parameters:
    - url: str : The Figshare download URL.
    - download_path: str : The path to the directory where the file should be downloaded and unzipped.
    """
    try:
        # Send HTTP request to the Figshare URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors

        # Get the content disposition header to determine the filename
        content_disposition = response.headers.get('content-disposition')
        if content_disposition:
            filename = content_disposition.split('filename=')[-1].strip('"')
        else:
            filename = os.path.basename(url)
        
        file_path = os.path.join(download_path, filename)

        # Download the file in chunks and save it to the specified path
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Downloaded file to {file_path}")

        # Check if the file is a zip file
        if zipfile.is_zipfile(file_path):
            # Unzip the file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print(f"Unzipped file to {download_path}")
            
            # Remove the zip file
            os.remove(file_path)
            print(f"Removed zip file {file_path}")
        else:
            print(f"Downloaded file is not a zip file: {file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error unzipping file: {str(e)}")

def download_a_file_from_url(url, download_path='.'):
    """
    Downloads a file from a URL.
    
    Parameters:
    - url: str : The URL of the file to download.
    - download_path: str : The path to the directory where the file should be downloaded.
    """
    try:
        # Send HTTP request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors

        # Get the filename from the URL
        filename = os.path.basename(url)
        file_path = os.path.join(download_path, filename)

        # Download the file in chunks and save it to the specified path
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Downloaded file to {file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def download_and_extract_from_url(url, download_path='.'):
    """
    Downloads a zip or tgz file from a URL and extracts it.
    
    Parameters:
    - url: str : The URL of the file to download.
    - download_path: str : The path to the directory where the file should be downloaded.
    """
    try:
        # Send HTTP request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors

        # Get the filename from the URL
        filename = os.path.basename(url)
        file_path = os.path.join(download_path, filename)

        # Download the file in chunks and save it to the specified path
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Downloaded file to {file_path}")

        # Check if the file is a zip file
        if zipfile.is_zipfile(file_path):
            # Unzip the file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print(f"Unzipped zip file to {download_path}")
        
        # Check if the file is a tar.gz or tgz file
        elif tarfile.is_tarfile(file_path):
            # Extract the tar file
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(download_path)
            print(f"Extracted tgz file to {download_path}")
        
        else:
            print(f"Downloaded file is neither a zip nor a tgz file: {file_path}")
            return
        
        # Remove the downloaded file after extraction
        os.remove(file_path)
        print(f"Removed downloaded file {file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading zip file: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error unzipping file: {str(e)}")
        
def download_dataset_from_tensorflow(file, download_path='.'):
    """
    Downloads a dataset from TensorFlow Datasets.
    
    Parameters:
    - file: str : The name of the dataset file to download.
    - download_path: str : The path to the directory where the dataset should be downloaded.
    """
    try:
        # Load the dataset
        dataset = tfds.load(file, data_dir=download_path)
        print(f"Downloaded dataset to {download_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
    


def download_dataset_huggingface(dataset_name, download_path='.'):
    """
    Downloads a dataset from Hugging Face and saves each subset as a CSV file.
    
    Parameters:
    - dataset_name: str : The name of the dataset on Hugging Face.
    - download_path: str : The path to the directory where the dataset CSVs should be saved.
    """
    # Define the dataset name (without subset/config)
    # dataset_name = "toxigen/toxigen-data"

    # Get the list of available configurations (subsets) for this dataset
    configs = get_dataset_config_names(dataset_name)

    # Iterate over each subset/config and load it
    try:
        for config in configs:
            # Load the dataset from Hugging Face
            dataset = load_dataset(dataset_name,config)
            
            # Iterate over each subset in the dataset (e.g., train, test, validation)
            for subset_name in dataset:
                # Get the subset
                subset = dataset[subset_name]
                
                # Save the subset as a CSV file
                subset.to_csv(f'{download_path}/{dataset_name}_{config}_{subset_name}.csv')
                print(f"Saved {subset_name} subset as CSV to {download_path}/{dataset_name}_{subset_name}.csv")
        
    except Exception as e:
        print(f"An error occurred: {e}")  

def download_huggingface_dataset_no_config(dataset_name, download_path='.'):
    snapshot_download(repo_id=dataset_name,local_dir=download_path,repo_type="dataset")

if __name__ == '__main__':

    project_path=os.path.abspath(__file__).split('src')[0]
    
    download_a_file_from_url("https://raw.githubusercontent.com/sabithsn/APPDIA-Discourse-Style-Transfer/refs/heads/main/data/original-annotated-data/original-dev.tsv", download_path=project_path+'data/raw/APPDIA')
    download_a_file_from_url("https://raw.githubusercontent.com/sabithsn/APPDIA-Discourse-Style-Transfer/refs/heads/main/data/original-annotated-data/original-test.tsv", download_path=project_path+'data/raw//APPDIA')
    download_a_file_from_url("https://raw.githubusercontent.com/sabithsn/APPDIA-Discourse-Style-Transfer/refs/heads/main/data/original-annotated-data/original-train.tsv", download_path=project_path+'data/raw//APPDIA')
    download_a_file_from_url("https://raw.githubusercontent.com/s-nlp/paradetox/refs/heads/main/paradetox/paradetox.tsv", download_path=project_path+'data/raw//paradetox')
    download_a_file_from_url("https://raw.githubusercontent.com/s-nlp/parallel_detoxification_dataset/refs/heads/main/parallel_detoxification_dataset_small.tsv", download_path=project_path+'data/raw//parallel_detoxification_dataset')
    download_dataset_huggingface(dataset_name="TheMrguiller/toxicity_big_bird", download_path=project_path+'data/raw//toxicity_dataset')
    download_a_file_from_url("https://raw.githubusercontent.com/anirudhsom/CAPP-Dataset/refs/heads/main/Generated_Paraphrases/APPDIA_Generated-Paraphrases.csv",download_path=project_path+'data/external/CAPP_article')
    download_a_file_from_url("https://raw.githubusercontent.com/anirudhsom/CAPP-Dataset/refs/heads/main/Generated_Paraphrases/ParaDetox_Generated-Paraphrases.csv",download_path=project_path+'data/external/CAPP_article')