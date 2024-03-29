import torch
import torchvision

from PIL import Image


class CIFAR10_captioning(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.vocab = ['An', "image", "of", "a",
                      *self.classes, '<SOS>', '<EOS>']
        self.token2idx = {t: i for i, t in enumerate(self.vocab)}

    def __getitem__(self, index):
        img, target_id = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        target_cls = self.classes[target_id]
        target = torch.tensor([self.token2idx[token]
                               for token in ['<SOS>', 'An', 'image', 'of', 'a', target_cls, '<EOS>']])
        
        return img, (target, target_id)
