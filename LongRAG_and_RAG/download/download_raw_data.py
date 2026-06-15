import gdown
import os
import zipfile
import shutil

# Google Drive direct-download URL.
url = "https://drive.google.com/uc?export=download&id=1ox0T940B_9_pZquKFu7oTqWJPeiYXaoA"
output = 'data.zip'

# Download the archive.
gdown.download(url, output, quiet=False)

# Extract the archive.
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(path='..', pwd=None)
    # Remove platform-specific metadata files.
    for root, dirs, files in os.walk('..'):
        for file in files:
            if file == ".DS_Store":
                os.remove(os.path.join(root, file))
        for dir in dirs:
            if dir == "__MACOSX":
                shutil.rmtree(os.path.join(root, dir))

# Remove the downloaded archive.
os.remove(output)
