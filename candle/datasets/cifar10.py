from torchvision.datasets import CIFAR10

class CIFAR10T(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10T, self).__init__(root, train, transform, target_transform,
                 download)
        self.data=  self.data.transpose(0,3,1,2)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target