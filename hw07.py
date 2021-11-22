from torch import nn, optim, from_numpy

import numpy as np

xy = np.loadtxt('data/diabetes.csv.gz', delimiter=',', dtype=np.float32)

x_data = from_numpy(xy[:500, 0:-1])
y_data = from_numpy(xy[:500, [-1]])

x_data_test = from_numpy(xy[500:, 0:-1])
y_data_test = from_numpy(xy[500:, [-1]])


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.l1_2_3 = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU()
        )

        self.l4_5_6 = nn.Sequential(
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
        )

        self.l7_8_9_10 = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.l1_2_3(x)
        x = self.l4_5_6(x)
        x = self.l7_8_9_10(x)

        return x


model = Model()

criterion = nn.BCELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch == 99:
        print(f"{epoch + 1} {loss.item():.4f}")

# for test
test_y_pred = model(x_data_test)
test_loss = criterion(test_y_pred, y_data_test)
print(f"Test {test_loss.item():.4f}")
