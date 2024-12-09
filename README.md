# ml-demos
Demos for AI/ML/DL/Generate AI while learning


## static check

Install the hooks of `pre-commit`
```shell
pre-commit install
```
Each time a commit is make, `pre-commit` will automatically run the configured tools.

OR

You cna manually run the following tools.
```shell
flake8 ml-demos/
pylint ml-demos/
mypy ml-demos/
black --check ml-demos/
isort --check-only ml-demos/
bandit -r ml-demos/
```
