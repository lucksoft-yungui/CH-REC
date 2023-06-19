import torch
import torchvision.transforms as transforms

class ZerosPAD(object):
    def __init__(self, max_size):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size

    def __call__(self, img):
        img = self.toTensor(img)
        c, h, w = img.shape
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad

        return Pad_img


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = \
                img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):
    def __init__(self, imgH=48, PAD='NormalizePAD'):
        self.imgH = imgH
        self.PAD = PAD

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        maxW = 0
        for image in images:
            h, w, c = image.shape
            if w > maxW:
                maxW = w

        if self.PAD == 'ZerosPAD':
            trans = ZerosPAD((1, self.imgH, maxW))
        elif self.PAD == 'NormalizePAD':
            trans = NormalizePAD((1, self.imgH, maxW))
        else:
            raise ValueError("not expected padding.")

        padded_images = []
        for image in images:
            h, w, c = image.shape
            padded_images.append(trans(image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in padded_images], 0)

        return image_tensors, labels