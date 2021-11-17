import torch
import torchvision

from PIL import Image
from pathlib import Path

from tqdm import tqdm
from pathlib import Path

from dataset import SCUT_FBP5500
from facenet_pytorch import MTCNN

DEVICE = torch.device('cpu')

PROJECT_ROOT = Path("/home/goda/face_beauty_prediction/")


#####


def evaluate(imgs, transform, model):
    for img_path in imgs:
        img = Image.open(PROJECT_ROOT / img_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        score = model(img)
        print(score.item())


if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        MTCNN(image_size=224),
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = torchvision.models.resnet18()
    fc_in = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in, 1)

    # WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/resnet18/60.pth"
    WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/resnet18_detection/33.pth"
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.to(device=DEVICE)
    model.eval()

    imgs = ['img01.jpg', 'img02.jpg', 'img03.jpg', 'img04.jpg', 'img05.jpg']
    # imgs = ['img12.jpg', 'img13.jpg', 'img14.jpg', 'img15.jpg']

    evaluate(imgs, transform, model)
