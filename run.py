from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import math
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    keep_all=False, device=device
)
dataset = datasets.ImageFolder('./photos')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=4)

embedding_db = 'embeddings.pt'

if os.path.exists(embedding_db):
    data = torch.load(embedding_db)
    names = data[0]
    embeddings = data[1]
else:
    print('Generating {} ...'.format(embedding_db))
    faces = []
    names = []
    for x, y in loader:
        face, prob = mtcnn(x, return_prob=True)
        if face is not None:
            print('Face detected with probability: {:8f}, {}'.format(prob, dataset.idx_to_class[y]))
            faces.append(face)
            names.append(dataset.idx_to_class[y])

    faces = torch.stack(faces).to(device)
    embeddings = resnet(faces).detach().cpu()

    torch.save([names, embeddings], embedding_db)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    keep_all=True, device=device
)
cap = cv2.VideoCapture(0)

print('Start recognizing ...')

while True:
    ret, frame = cap.read()
    if not ret:
        print("fail to read frame, try again")
        break

    img = Image.fromarray(frame)
    faces, probs = mtcnn(img, return_prob=True)

    if faces is not None:
        face_embeddings = resnet(faces.to(device)).detach().cpu()
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(probs):
            if prob > 0.90:
                face_embedding = face_embeddings[i]
                distances = []
                for embedding in embeddings:
                    distances.append(torch.dist(face_embedding[i], embedding).item())
                min_distance = min(distances)
                min_idx = distances.index(min_distance)
                name = names[min_idx] if min_distance < 1.5 else '???'

                box = boxes[i]
                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])) , (int(box[2]), int(box[3])), (255,0,0), 2)
                #frame = cv2.putText(frame, name + ' ' + str(min_distance), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                frame = cv2.putText(frame, name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
