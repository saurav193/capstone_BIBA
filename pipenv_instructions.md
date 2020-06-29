# General pipenv instructions

Adapted from the document on Google Drive

To install pipenv, run:

`pip install pipenv`.

To install a new library or package, use `pipenv` rather than `pip`. 

For example, run `pipenv install numpy` instead of `pip install numpy`.

To open a shell within the virtual environment, run:

`pipenv shell`.

Afterwards, you can run `jupyter notebook` or a python file.

After pulling a new branch or repository that has pipenv, run

`pipenv install`

to install all the packages that are required for the project.

Before pushing your code, run

`pipenv lock`

to create the equivalent of a `requirement.txt` file.

### NOTE

You may run into an issue where the modules are installed, but they aren’t importing into your Jupyter notebook.

Usually you would run these commands:

`pipenv shell`

`jupyter notebook`

Instead, run this:

`pipenv shell`

`python -m ipykernel install --user --name=capstone_machine_learning`

`jupyter notebook`
