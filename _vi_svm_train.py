# requirements:
# pip install facenet-pytorch sklearn numpy

from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
import os
from sklearn.svm import SVC
import joblib
import torch

# --- Initialize detectors/encoders ---
mtcnn = MTCNN(image_size=160, margin=0)  # detection + alignment
resnet = InceptionResnetV1(pretrained='vggface2').eval()  # embedding model

def collect_embeddings(people, DIR):
    embeddings = []
    labels = []
    for person in people:
        pdir = os.path.join(DIR, person)
        if not os.path.isdir(pdir):
            print(f"⚠️ Skipping missing directory: {pdir}")
            continue
        for fname in os.listdir(pdir):
            img_path = os.path.join(pdir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Failed to read image: {img_path}")
                continue
            img_rgb = img[:, :, ::-1]  # BGR → RGB
            face = mtcnn(img_rgb)
            if face is None:
                print(f"⚠️ No face detected in: {img_path}")
                continue
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0))  # 1 x 512
            emb_np = emb.cpu().numpy().reshape(-1)
            embeddings.append(emb_np)
            labels.append(person)  # ✅ use name string as label
    return np.array(embeddings), np.array(labels)

def main():
# --- Training ---
    people = ['Jerry', 'Madonna', 'Mindy', 'Vidura']
    # DIR = r'C:\Users\desmond\OneDrive\Resources\Faces\cropped'  # EITHER CROPPED
    DIR = r'C:\Users\Vidura\Documents\PSB academy\Visual interfaces\week8\Faces\Faces\train'      # OR TRAINED
    embs, labs = collect_embeddings(people, DIR)

    if len(embs) == 0:
        raise RuntimeError("❌ No embeddings collected. Check image paths and face detection.")

    clf = SVC(kernel='linear', probability=True)
    clf.fit(embs, labs)
    # joblib.dump(clf, './OCV_data/face_svm.pkl')               # EITHER CROPPED
    joblib.dump(clf, './OCV_data/face_svm_uncropped2.pkl')       # OR TRAINED
    print("✅ SVM model trained and saved to './OCV_data/face_svm.pkl'")

if __name__ == "__main__":
    main()

