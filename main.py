from torch.utils.data import DataLoader

from scripts.data import CILDataset

if __name__ == '__main__':
    # path to the given training data
    dataset = CILDataset('./data/monodepth_kaggle2026/train')

    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(10):
        for idx, (image, gt) in enumerate(dataloader):
            pass