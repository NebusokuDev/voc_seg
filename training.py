from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ToTensor

from transforms import ImgToTensor

if __name__ == '__main__':

    # VOCデータセットの取得
    voc = VOCSegmentation("./data", image_set="train", transform=ImgToTensor(), target_transform=ImgToTensor())

    # 画像とマスクを取得
    image, mask = voc[0]  # 先頭のサンプルを取

    print(image.shape)
    print(mask[0].max().item())