from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist


class MnistClassifier:
    def __init__(self) -> None:
        self.layers: List[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5),
            Tensor.relu,
            nn.Conv2d(32, 32, 5),
            Tensor.relu,
            nn.BatchNorm(32),
            Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3),
            Tensor.relu,
            nn.Conv2d(64, 64, 3),
            Tensor.relu,
            nn.BatchNorm(64),
            Tensor.max_pool2d,
            lambda x: x.flatten(1),
            nn.Linear(576, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = mnist()

    model = MnistClassifier()

    optimizer = nn.optim.AdamW(nn.state.get_parameters(model))

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        optimizer.zero_grad()
        samples: Tensor = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
        loss = (
            model(X_train[samples])
            .sparse_categorical_crossentropy(Y_train[samples])
            .backward()
        )
        optimizer.step()
        return loss

    @TinyJit
    def get_acc_test() -> Tensor:
        return (model(X_test).argmax(axis=1) == Y_test).mean() * 100

    test_acc = float("nan")
    for i in (t := trange(getenv("STEPS", 70))):
        GlobalCounters.reset()
        loss = train_step()
        if i % 10 == 9:
            test_acc = get_acc_test().item()
        t.set_description(f"loss: {loss.item():6.2f} test accuracy: {test_acc:5.2f}%")

        if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
            if test_acc >= target and test_acc != 100.0:
                print(colored(f"{test_acc} >= {target}", "green"))
            else:
                raise ValueError(colored(f"{test_acc} < {target}", "red"))
