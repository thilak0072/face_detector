# Face Clustering API

This FastAPI-based application enables users to upload images, perform face clustering using DBSCAN, and download the clustered faces in ZIP format.

## Features
- Upload multiple images for face clustering
- Detect and encode faces using `face_recognition`
- Cluster faces using `DBSCAN`
- Generate and serve ZIP files containing clustered faces
- Display a montage of clustered faces

## Requirements
Ensure you have the following dependencies installed before running the application:
```bash
pip install fastapi uvicorn numpy opencv-python face-recognition scikit-learn imutils
```

## Project Structure
- `main.py` - The FastAPI application
- `static/` - Directory for serving static files (ZIP archives)
- `uploads/` - Directory for storing uploaded images

## Endpoints
### 1. Upload Images and Perform Face Clustering
**Endpoint:** `POST /upload-images/`

**Description:**
- Accepts multiple image files as input.
- Saves uploaded images to `uploads/`.
- Detects faces and extracts 128-d encodings.
- Clusters faces using `DBSCAN`.
- Creates ZIP files containing images of clustered faces.
- Returns JSON response with unique face count, ZIP file links, and montages.

**Example Request (cURL):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/upload-images/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@image1.jpg' \
  -F 'files=@image2.jpg'
```

### 2. Download Clustered Faces ZIP File
**Endpoint:** `GET /download/{file_name}`

**Description:**
- Allows downloading of ZIP files containing clustered faces.
- Example: `http://127.0.0.1:8000/download/zip_face#1.zip`

### 3. Web Interface (Homepage)
**Endpoint:** `GET /`

**Description:**
- Provides an HTML form for users to upload images via a browser.

## How It Works
1. **Face Detection & Encoding**
   - Uses `face_recognition.face_locations` to detect faces.
   - Extracts 128-dimensional encodings using `face_recognition.face_encodings`.

2. **Face Clustering**
   - Uses `DBSCAN` for unsupervised clustering based on face encodings.
   - Groups similar faces into clusters and assigns unique labels.

3. **Storing & Serving Clustered Faces**
   - Images of clustered faces are saved in folders.
   - ZIP archives are created for each cluster.
   - The response includes links to download the ZIP files.

## Running the Application
Start the FastAPI server using:
```bash
uvicorn main:app --reload
```
Access the web interface at: `http://127.0.0.1:8000/`

# Face Clustering using Streamlit and DBSCAN

## Overview
This application clusters faces from uploaded images using **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** and **face_recognition**. The clustered images are displayed in an interactive UI, and users can download zip files of the clustered faces.

## Features
- Upload multiple images containing faces.
- Detect and encode faces using **face_recognition**.
- Cluster similar faces using **DBSCAN**.
- Display detected faces in a user-friendly **Streamlit** interface.
- Allow users to download the clustered images as zip files.

## Requirements
Before running the application, ensure you have the following dependencies installed:

```bash
pip install streamlit numpy opencv-python face-recognition imutils scikit-learn
```

Additionally, you may need to install `dlib` if it's not already installed:

```bash
pip install dlib
```

## How to Run the Application
1. Clone this repository or copy the script to your local machine.
2. Install the required dependencies using the above commands.
3. Run the Streamlit application using the following command:

```bash
streamlit run app.py
```

4. Upload images containing faces through the web interface.
5. The application will process the images, detect faces, and cluster similar faces.
6. View the clustered faces in the UI and download them as zip files.

## Explanation of the Code

1. **Importing Dependencies:**
   - Necessary libraries such as **Streamlit**, **OpenCV**, **face_recognition**, and **DBSCAN** are imported.
   
2. **Setting up Streamlit UI:**
   - A web-based interface is created using **Streamlit**.
   - Users can upload multiple images for processing.
   
3. **Processing Uploaded Images:**
   - Images are temporarily stored and processed using **OpenCV**.
   - The application detects faces using `face_recognition.face_locations()`.
   - Encodings for each face are generated using `face_recognition.face_encodings()`.
   
4. **Clustering Faces:**
   - The **DBSCAN** clustering algorithm is used to group similar faces.
   - Each unique face is assigned a label, while unknown faces are labeled separately.
   
5. **Displaying Results:**
   - The identified clusters are displayed in the UI.
   - A montage of detected faces is generated using `imutils.build_montages()`.
   - Users can expand sections to view all images corresponding to a face cluster.
   
6. **Downloading Clustered Faces:**
   - Clustered images are saved into separate folders.
   - These folders are zipped and made available for download.

## Example Usage
- Upload images with multiple people.
- The system will detect, encode, and cluster similar faces.
- View and download clustered images for further analysis.


