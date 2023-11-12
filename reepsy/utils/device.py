import torch


class Device():
    """

    Класс для автоматического определения девайса

    """
    def get_device() -> type[torch.device]:
        """
        Метод для автоматического определения девайса

        Возвращает "cpu" или "cuda"
        """
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
