from reepsy.core.network import Network
from reepsy.core.data import Data
from reepsy.core.classification import Classification


network = Network()

train_dataset = Data.load_dataset(
    dataset_data_path='./dataset/train',
    dataset_csv_path='./dataset/train.csv',
)

test_dataset = Data.load_dataset(
    dataset_data_path='./dataset/test',
    dataset_csv_path='./dataset/test.csv',
)

model = network.train(dataset=train_dataset, num_epochs=2)
predict = network.predict(dataset=test_dataset)

print(f'Predictions: {predict}')

index = Classification.classify_picture(
    model=model,
    picture_path='./dataset/test/q.20.jpg'
)

print(f'Picture class: {index}')
