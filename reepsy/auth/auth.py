import cv2

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.models.googlenet import googlenet
from torchvision.transforms import ToTensor, Compose
from torchvision import transforms

from ..dataset.dataset import DatasetLoader


class Auth():
    '''

    ТУТ БУДЕТ КАКОЙ-ТО БОЛЬШОЙ КОММЕНТАРИЙ

    TODO: переорганизовать логику, разделить на auth-network и на сам auth только переименовать
    добавить метод для

    

    '''

    def __init__(
        self,
        model=googlenet,
        criterion=CrossEntropyLoss,
        optimizer=Adam,
        transform=ToTensor,
        learning_rate: float = 1e-3
    ) -> None:
        self.__device = self.__get_device()
        self.__model = model(pretrained=True).to(self.__device)
        self.__criterion = criterion()
        self.__optimizer = optimizer(self.__model.parameters(), learning_rate)
        self.__transform = transform()

    def __get_device(self) -> type[torch.device]:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_dataset(
        self,
        dataset_data_path: str = './dataset/data',
        dataset_csv_path: str = './dataset/csv.csv',
        batch_size: int = 32
    ) -> DataLoader:
        dataset = DatasetLoader(
            dataset_csv_path=dataset_csv_path,
            dataset_data_path=dataset_data_path,
            transform=self.__transform
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return loader

    def trained_model(self, dataset, num_epochs: int = 1) -> None:
        for epoch in range(num_epochs):
            losses = []

            for _, (data, targets) in enumerate(dataset):
                data = data.to(device=self.__device)
                targets = targets.to(device=self.__device)

                scores = self.__model(data)
                loss = self.__criterion(scores, targets)

                losses.append(loss.item())

                self.__optimizer.zero_grad()
                loss.backward()

                self.__optimizer.step()
            print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')

    def check_accuracy(self, dataset) -> None:
        num_correct = 0
        num_samples = 0
        self.__model.eval()

        with torch.no_grad():
            for x, y in dataset:
                x = x.to(device=self.__device)
                y = y.to(device=self.__device)

                scores = self.__model(x)
                _, predirections = scores.max(1)
                num_correct += (predirections == y).sum()
                num_samples += predirections.size(0)

            print(
                f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}%')
        self.__model.train()

    def get_person(self, signature_image_path) -> int:
        image = cv2.imread(signature_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = Compose([self.__transform])
        with torch.no_grad():
            self.__model.to(self.__device)
            self.__model.eval()

            tensor = transform(image)
            tensor = tensor.to(self.__device)

            scores = self.__model(tensor.unsqueeze(0))
            _, [index] = scores.max(1)

            return int(index)
