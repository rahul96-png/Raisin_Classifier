import urllib.request
import zipfile
import io
import pandas as pd
from scipy.io import arff
import os

def download_and_extract_dataset():
    url = "https://archive.ics.uci.edu/static/public/850/raisin.zip"
    csv_filename = "raisin.csv"
    
    print(f"Downloading Raisin dataset from {url}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
        
        print("Dataset downloaded. Extracting zip files...")
        with zipfile.ZipFile(io.BytesIO(zip_data)) as outer_z:
            nested_zip_data = outer_z.read("Raisin_Dataset.zip")
            
        with zipfile.ZipFile(io.BytesIO(nested_zip_data)) as inner_z:
            arff_content = inner_z.read("Raisin_Dataset/Raisin_Dataset.arff")
            
        print("Parsing ARFF data...")
        data, meta = arff.loadarff(io.StringIO(arff_content.decode('utf-8')))
        df = pd.DataFrame(data)
        
        # Clean target column if it's byte strings
        if df['Class'].dtype == object:
            df['Class'] = df['Class'].str.decode('utf-8')
            
        print(f"Saving data to {csv_filename}...")
        df.to_csv(csv_filename, index=False)
        print("Success! Dataset downloaded and saved.")
        
    except Exception as e:
        print(f"Error downloading or processing dataset: {e}")

if __name__ == "__main__":
    download_and_extract_dataset()
