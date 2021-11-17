import torch
import torchvision

from tqdm import tqdm
from pathlib import Path

from dataset import SCUT_FBP5500_MAN
from facenet_pytorch import MTCNN

#####
THRESHOLD = 15
NUM_EPOCH = 200
BATCH_SIZE = 16

LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)

STEP_SIZE = 20
GAMMA = 0.5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = 20

PROJECT_ROOT = Path("/home/goda/face_beauty_prediction/")
# WEIGHTS_PATH = PROJECT_ROOT / "pretrained_weights/resnext50_32x4d/52.pth"
#####

MIN_TEST_LOSS = float("inf")


def train(model, optimizer, scheduler, mse, mae, train_loader):
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    model.train()

    for batch_index, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        scores = model(imgs)

        loss = mse(scores, labels)
        loss_sub = mae(scores, labels)

        total_mse += loss
        total_mae += loss_sub
        total_count += 1

        # loss.backward()
        loss_sub.backward()
        optimizer.step()

    scheduler.step()

    return total_mse, total_mae, total_count


def evaluate(model, mse, mae, test_loader):
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    model.eval()

    with torch.no_grad():
        for batch_index, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            scores = model(imgs)

            loss = mse(scores, labels)
            loss_sub = mae(scores, labels)

            total_mse += loss
            total_mae += loss_sub
            total_count += 1

    return total_mse, total_mae, total_count


def logging(train_result, test_result):
    def log(total_mse, total_mae, total_count):
        return f"{total_mse / total_count:.6f} {total_mae / total_count:.6f} "

    return log(*train_result) + log(*test_result)


if __name__ == "__main__":

    transform_for_train = torchvision.transforms.Compose([
        MTCNN(image_size=224, post_process=True),
    ])

    transform_for_test = torchvision.transforms.Compose([
        MTCNN(image_size=224, post_process=True),
    ])

    # transform_for_train = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(256),
    #     torchvision.transforms.CenterCrop(224),
    #     # torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    #
    # transform_for_test = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(256),
    #     torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    train_dataset = SCUT_FBP5500_MAN(
        is_train=True,
        transform=transform_for_train
    )

    test_dataset = SCUT_FBP5500_MAN(
        is_train=False,
        transform=transform_for_test
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
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    model = torchvision.models.resnet50()
    fc_in = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in, 1)
    model.to(device=DEVICE)

    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    miss = 0
    for epoch in tqdm(range(NUM_EPOCH)):
        train_result = train(model=model,
                             optimizer=optimizer, scheduler=scheduler,
                             mse=mse, mae=mae,
                             train_loader=train_loader)

        test_result = evaluate(model=model,
                               mse=mse, mae=mae,
                               test_loader=test_loader)

        log = logging(train_result, test_result)
        print(epoch, log)

        if test_result[1] < MIN_TEST_LOSS:
            MIN_TEST_LOSS = test_result[1]
            miss = 0
            torch.save(model.state_dict(),
                       PROJECT_ROOT / 'pretrained_weights' / 'resnet50_MAE_detection_man_ext' / f"{epoch}.pth")
        else:
            miss += 1

        if miss >= THRESHOLD:
            break
