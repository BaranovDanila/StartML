from torch.nn import Sequential, ReLU, Linear


def create_model():
    model = Sequential(
        Linear(100, 10),
        ReLU(),
        Linear(10, 1)
    )

    return model


# result = create_model()
# print(list(result.parameters()))
