from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor

from reepsy.utils.preparing_dataset import PreparingDataset


class Data():
    """

    Класс для создания датасета

    """
    def load_dataset(dataset_data_path: str, dataset_csv_path: str, transform=ToTensor(),  batch_size: int = 32) -> DataLoader:
        """
        Метод создания датасета
            - dataset_data_path - путь к директории с изображениями
            - dataset_csv_path - путь к csv фалу
            - transform - объект класса трансформации изображения из PyTorch, по стандарту ToTensor()
            - batch_size - количество образцов данных, которые одновременно подаются на вход нейронной сети во время одной итерации обучения, по стандарту 32

        Возвращаемое значение: индетификатор объекта
        """
        preparing_dataset = PreparingDataset(
            dataset_csv_path=dataset_csv_path,
            dataset_data_path=dataset_data_path,
            transform=transform
        )
        dataset = DataLoader(
            dataset=preparing_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return dataset
