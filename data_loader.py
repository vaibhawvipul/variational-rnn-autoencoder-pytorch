import torch
from torchvision import datasets, transforms

def get_loaders(batch_size, num_workers = 0):
    trans = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST('./data', train=True, transform=trans, download=True)
    testset = datasets.MNIST('./data', train=False, transform=trans, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

if __name__ == "__main__":
    trainloader, testloader = get_loaders(batch_size=4)

    print(trainloader.dataset.classes)
