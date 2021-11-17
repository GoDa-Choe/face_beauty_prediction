import torch
import torchvision

from PIL import Image
from pathlib import Path

from tqdm import tqdm
from pathlib import Path

from dataset import SCUT_FBP5500

#####
NUM_EPOCH = 300
BATCH_SIZE = 1

LEARNING_RATE = 0.01
BETAS = (0.9, 0.999)

STEP_SIZE = 40
GAMMA = 0.5

DEVICE = torch.device('cpu')
NUM_WORKERS = 4

PROJECT_ROOT = Path("/home/goda/face_beauty_prediction/")


# WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/A_resnet18/63.pth"
# WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/AM_resnet18/74.pth"
# WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/AM_FULL_resnet18/59.pth"
# WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/resnext50_32x4d/46.pth"
# WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/efficientnet_b4/21.pth"


#####

def evaluate(model, test_loader):
    model.eval()
    for image, label in test_loader:
        image, label = image.to(DEVICE), label.to(DEVICE)
        print(image.shape)
        score = model(image)

        print("scores", score.item())
        print("labels", label.item())


if __name__ == "__main__":
    img = Image.open(PROJECT_ROOT / 'img15.jpg').convert('RGB')
    img.show()

    resize_crop = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(500),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
    ])

    tensor_normal = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = resize_crop(img)

    img = tensor_normal(img)
    img = img.unsqueeze(0)

    model = torchvision.models.resnet18()

    fc_in = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in, 1)

    # model = torchvision.models.efficientnet_b4()
    # model.classifier = torch.nn.Sequential(
    #     torch.nn.Dropout(p=0.4, inplace=True),
    #     torch.nn.Linear(in_features=1792, out_features=1, bias=True)
    # )

    # WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/resnet18_ext/25.pth"
    WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/resnet18/60.pth"

    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.to(device=DEVICE)
    # evaluate(model=model, test_loader=test_loader)
    model.eval()
    score = model(img)
    print(score.item())
