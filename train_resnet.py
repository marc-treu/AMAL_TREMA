import torch
import torch.nn as nn

from models.ResNet import CifarResNeXt
from dataset.CIFAR10_Dataset import get_CIFAR


print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


epochs = 30
batch_size = 128
learning_rate = 0.05
momentum = 0.9
decay = 0.0005


trainloader, testloader, mean, std, classes = get_CIFAR(data_augmentation=True, batch_size=batch_size)

ResNet = CifarResNeXt(8, 20, 10, 64, 4)
ResNet.to(device)

# ResNet.load_state_dict(torch.load('pretrained_models/ResNet_cifar10.cpkt', map_location=device))


optimizer = torch.optim.SGD(ResNet.parameters(), learning_rate, momentum, weight_decay=decay, nesterov=True)
criterion = nn.CrossEntropyLoss()


c1_train_loss = []
c1_test_loss = []
c1_train_prec = []
c1_test_prec = []
c1_mean_train_loss = []
c1_mean_test_loss = []

nb_train = len(trainloader) * batch_size
nb_test = len(testloader) * batch_size

best_state_dict = ([], 0.0)

print("Training Begin")
for ep in range(epochs):

    sum_train = 0
    sum_test = 0
    c1_train_loss = []
    c1_test_loss = []
    print("\n\nepoch:", ep)
    ResNet.train()

    for i, (x, y) in enumerate(trainloader):

        y = y.to(device)
        output = ResNet(x.to(device))

        optimizer.zero_grad()
        loss = criterion(output, y)

        c1_train_loss.append(float(loss))
        sum_train += int(torch.sum(torch.argmax(output, 1) == y))

        loss.backward()
        optimizer.step()

    c1_mean_train_loss.append(sum(c1_train_loss) / len(c1_train_loss))
    print("train : loss =", round(c1_mean_train_loss[-1], 6), 'precision =', round(sum_train / nb_train, 3))
    c1_train_prec.append(sum_train / nb_train)

    ResNet.eval()

    for i, (x, y) in enumerate(testloader):
        with torch.no_grad():
            y = y.to(device)
            output = ResNet(x.to(device))
            loss = criterion(output, y)

            # accuracy
            c1_test_loss.append(float(loss))
            sum_test += int(torch.sum(torch.argmax(output, 1) == y))

    c1_mean_test_loss.append(sum(c1_test_loss) / len(c1_test_loss))
    print("test : loss =", round(c1_mean_test_loss[-1], 6), 'precision =', round(sum_test / nb_test, 3))
    c1_test_prec.append(sum_test / nb_test)

    if c1_test_prec[-1] > best_state_dict[1]:
        best_state_dict = (ResNet.state_dict(), c1_test_prec[-1])

    if len(c1_mean_test_loss) > 3 and c1_mean_test_loss[-1] > sum(c1_mean_test_loss[-3:-1]) / 2:
        print("End training")
        torch.save(best_state_dict[0], 'ResNet_train_final.cpkt')
        print('ResNet save, ResNet_train_final.cpkt')
        break
