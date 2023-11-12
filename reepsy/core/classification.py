import cv2
import torch

from torchvision.transforms import ToTensor, Compose

from reepsy.utils.device import Device


class Classification():
    """

    Класс для классификации изображений

    """

    def classify_picture(model: any, picture_path: str, device: Device = Device.get_device(), transform=ToTensor()) -> int:
        """
        Метод классификации изображения
            - model - обученная модель из Reepsy (или PyTorch)
            - picture_path - путь к классифицируемому изображению
            - device - на чем будут производится вычисления, по стандарту определяется автоматически
                - "cpu"
                - "cuda"
            - transform - объект класса трансформации изображения из PyTorch, по стандарту ToTensor()

        Возвращаемое значение: индетификатор объекта
        """
        picture = cv2.imread(picture_path)
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        transform = Compose([transform])
        with torch.no_grad():
            model.to(device)
            model.eval()

            tensor = transform(picture)
            tensor = tensor.to(device)

            scores = model(tensor.unsqueeze(0))
            _, [index] = scores.max(1)

        return int(index)
