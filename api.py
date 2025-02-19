import os
import shutil
import numpy as np
import cv2
import face_recognition
from sklearn.cluster import DBSCAN
from imutils import build_montages
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()


UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/upload-images/")
async def upload_images(files: list[UploadFile] = File(...)):
    """Handles image uploads and performs face clustering."""
    file_paths = []

    
    for file in files:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        file_paths.append(file_location)

    # Face encoding and clustering
    data = []
    for file_path in file_paths:
        image = cv2.imread(file_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")  # Faster than "cnn"
        encodings = face_recognition.face_encodings(rgb, boxes)

        for (box, enc) in zip(boxes, encodings):
            data.append({"imagePath": file_path, "loc": box, "encoding": enc})

    if not data:
        raise HTTPException(status_code=400, detail="No faces detected.")

    # Convert data into a numpy array
    encodings_arr = np.array([d["encoding"] for d in data])

    # Clustering with DBSCAN
    cluster = DBSCAN(min_samples=3, metric="euclidean").fit(encodings_arr)
    labelIDs = np.unique(cluster.labels_)
    num_unique_faces = len(labelIDs[labelIDs != -1])  # Ignore noise (-1)

    response = {"numUniqueFaces": num_unique_faces, "faces": []}

    # Process each cluster (skip labelID == -1 which represents noise)
    for labelID in labelIDs:
        if labelID == -1:
            continue

        idxs = np.where(cluster.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(15, len(idxs)), replace=False)
        faces = []

        dir_name = os.path.join(STATIC_DIR, f'face_group_{labelID + 1}')
        os.makedirs(dir_name, exist_ok=True)

        for i in idxs:
            data_point = data[i]
            image = cv2.imread(data_point["imagePath"])
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            (top, right, bottom, left) = data_point["loc"]
            face = rgb_image[top:bottom, left:right]
            face = cv2.resize(face, (96, 96))
            faces.append(face)

            
            face_image_path = os.path.join(dir_name, f"face_{i}.jpg")
            cv2.imwrite(face_image_path, image)

        
        zip_filename = f'face_group_{labelID + 1}.zip'
        zip_path = os.path.join(STATIC_DIR, zip_filename)
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', dir_name)
        shutil.rmtree(dir_name)  

    
        montage = build_montages(faces, (96, 96), (2, 2))[0] if faces else None

        response["faces"].append({
            "faceID": labelID + 1,
            "zipFile": f"/static/{zip_filename}",
            "montage": montage.tolist() if montage is not None else None
        })

    return response


@app.get("/download/{file_name}")
async def download_file(file_name: str):
    """Endpoint to download grouped face images as ZIP."""
    file_path = os.path.join(STATIC_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=file_name)


@app.get("/", response_class=HTMLResponse)
def home():
    """Basic HTML upload form."""
    return """
    <html>
        <head>
            <title>Face Clustering</title>
        </head>
        <body>
            <h1>Upload your images for face clustering</h1>
            <form action="/upload-images/" method="post" enctype="multipart/form-data">
                <input type="file" name="files" multiple>
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """
