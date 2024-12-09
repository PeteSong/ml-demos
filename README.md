# ml-demos
Demos for AI/ML/DL/Generate AI while learning


## static check
Install the static check tools.
```shell
pip install flake8 pylint mypy black isort bandit
```

Install the hooks of `pre-commit`
```shell
pre-commit install
```
Each time a commit is make, `pre-commit` will automatically run the configured tools.

OR

You cna manually run the following tools.
```shell
flake8 ml_demos/
pylint ml_demos/
mypy ml_demos/
black --check ml_demos/
isort --check-only ml_demos/
bandit -r ml_demos/
```
