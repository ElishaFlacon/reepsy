from reepsy.auth.auth import Auth

auth = Auth()

dataset = auth.load_dataset()

auth.trained_model(dataset, 2)
auth.check_accuracy(dataset)

index = auth.get_person('dataset/data/q.12.jpg')
print(index)
