import torch
import torchvision

from tqdm import tqdm
from pathlib import Path

from dataset import SCUT_FBP5500_MAN
from facenet_pytorch import MTCNN

#####
THRESHOLD = 30
NUM_EPOCH = 200
BATCH_SIZE = 16

LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)

STEP_SIZE = 20
GAMMA = 0.5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 16

PROJECT_ROOT = Path("/home/goda/face_beauty_prediction/")

MIN_TEST_LOSS = float("inf")


def blue(text):
    return '\033[94m' + text + '\033[0m'


def train(model, optimizer, criterion, scheduler, train_loader):
    total_loss = 0.0

    model.train()
    for batch_index, (imgs, labels) in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        scores = model(imgs)

        loss = criterion(scores, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    scheduler.step()

    return total_loss, batch_index


def evaluate(model, criterion, test_loader):
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for batch_index, (imgs, labels) in enumerate(test_loader, start=1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            scores = model(imgs)

            loss = criterion(scores, labels)
            total_loss += loss

    return total_loss, batch_index


def logging(train_result, test_result):
    def log(total_loss, total_count):
        return f"{total_loss / total_count:.6f} "

    return log(*train_result) + log(*test_result)


if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
        MTCNN(image_size=224, post_process=True),
    ])

    train_dataset = SCUT_FBP5500_MAN(
        is_train=True,
        transform=transform
    )

    test_dataset = SCUT_FBP5500_MAN(
        is_train=False,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model = torchvision.models.mobilenet_v2()
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(in_features=1280, out_features=1, bias=True),
    )

    model.to(device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    criterion = torch.nn.L1Loss()

    miss = 0
    for epoch in tqdm(range(NUM_EPOCH)):
        train_result = train(model=model, criterion=criterion, train_loader=train_loader,
                             optimizer=optimizer, scheduler=scheduler, )

        test_result = evaluate(model=model, criterion=criterion, test_loader=test_loader)

        log = logging(train_result, test_result)

        if test_result[0] < MIN_TEST_LOSS:
            MIN_TEST_LOSS = test_result[0]
            miss = 0
            torch.save(model.state_dict(),
                       PROJECT_ROOT / 'pretrained_weights' / 'mobilenetv2_man_long' / f"{epoch}.pth")
            print(epoch, blue(log))
        else:
            miss += 1
            print(epoch, log)

        if miss >= THRESHOLD:
            break
