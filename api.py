import os
import shutil
import tempfile
import numpy as np
import cv2
import face_recognition
from sklearn.cluster import DBSCAN
from imutils import build_montages
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import io
from zipfile import ZipFile


app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Endpoint to handle image uploads and perform face clustering
@app.post("/upload-images/")
async def upload_images(files: list[UploadFile] = File(...)):
    # Save uploaded files to the disk
    file_paths = []
    for file in files:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        file_paths.append(file_location)

    # Process uploaded images
    data = []
    for file_path in file_paths:
        image = cv2.imread(file_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": file_path, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        data.extend(d)

    # Converting the data into a numpy array
    data_arr = np.array(data)
    # Extracting the 128-d facial encodings
    encodings_arr = [item["encoding"] for item in data_arr]

    # Clustering with DBSCAN
    cluster = DBSCAN(min_samples=3)
    cluster.fit(encodings_arr)

    labelIDs = np.unique(cluster.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])

    # Prepare for returning the results
    response = {"numUniqueFaces": numUniqueFaces, "faces": []}

    for labelID in labelIDs:
        idxs = np.where(cluster.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(15, len(idxs)), replace=False)
        faces = []
        whole_images = []

        if labelID != -1:
            dir_name = f'face#{labelID + 1}'
            os.makedirs(dir_name, exist_ok=True)

        for i in idxs:
            current_image = cv2.imread(data_arr[i]["imagePath"])
            rgb_current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            (top, right, bottom, left) = data_arr[i]["loc"]
            current_face = rgb_current_image[top:bottom, left:right]
            current_face = cv2.resize(current_face, (96, 96))
            whole_images.append(rgb_current_image)
            faces.append(current_face)

            if labelID != -1:
                face_image_name = f'image{i}.jpg'
                cv2.imwrite(os.path.join(dir_name, face_image_name), current_image)

        if labelID != -1:
            zip_filename = f'zip_face#{labelID + 1}.zip'
            shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', dir_name)
            shutil.rmtree(dir_name) 

        
            response["faces"].append({
                "faceID": labelID + 1,
                "zipFile": f"/static/{zip_filename}" 
            })

        
        montage = build_montages(faces, (96, 96), (2, 2))[0]
        response["faces"][-1]["montage"] = montage.tolist()  

    return response

# Endpoint to serve a zip file (for downloading)
@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/")
def home():
    return """
    <html>
        <head><title>Face Clustering</title></head>
        <body>
            <h1>Upload your images for face clustering</h1>
            <form action="/upload-images/" method="post" enctype="multipart/form-data">
                <input type="file" name="files" multiple>
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

