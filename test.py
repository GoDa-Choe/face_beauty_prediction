import torch
import torchvision

from PIL import Image

from pathlib import Path

from facenet_pytorch import MTCNN

DEVICE = torch.device('cpu')

PROJECT_ROOT = Path("/home/goda/face_beauty_prediction/")


#####


def evaluate(imgs, transform, model):
    with torch.no_grad():
        for img_path in imgs:
            img = Image.open(PROJECT_ROOT / 'imgs' / img_path).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)
            score = model(img)
            print(f"{score.item():.2f}", end=" ")


if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        MTCNN(image_size=224, device=DEVICE),
    ])

    model = torchvision.models.resnet18()
    fc_in = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in, 1)
    # WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/resnet18_man/74.pth"
    WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/resnet18_man/60.pth"

    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.to(device=DEVICE)
    model.eval()

    imgs_good = ['img01.jpg', 'img02.jpg', 'img03.jpg', 'img04.jpg', 'img05.jpg', 'img06.jpg', 'img07.jpg']
    imgs_bad = ['img12.jpg', 'img13.jpg', 'img14.jpg', 'img15.jpg']
    imgs_mix = ['img18.jpg', 'img19.jpeg', 'img20.jpeg', 'img21.jpg', 'img22.jpeg']

    evaluate(imgs_good, transform, model)
    print()

    evaluate(imgs_bad, transform, model)
    print()

    evaluate(imgs_mix, transform, model)
    print()
