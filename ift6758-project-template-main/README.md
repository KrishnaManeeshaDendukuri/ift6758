# IFT6758 Project Codebase

* **ift6758-project-template-main/ift6758/data/**: Contains the modules for the questions 1, 2 and 4 i.e. to retreive, download and tidy the raw data from the NHL API.
* **ift6758-project-template-main/ift6758/visualizations/**: Contains the module for question 6 that deals with the advanced visualizations using plotly.
* **ift6758-project-template-main/notebooks/**: Contains the notebooks for individual questions from 1 to 6, which in turn make use of the modules written in the [ift6758/data/](https://github.com/etiennedemers/ift6758/tree/master/ift6758-project-template-main/ift6758/data) folder.

**Note:** 
1. This section assumes that one is running the codebase from the repository folder.
2. Code for Question 2 has been included via a module in ift6758/data/ folder and logic discussed in the blog
* **Usage**
```python
result_file = get_season_files(seasons = [2019], clear_existing_files = False)   
```
4. Taking the order of questions into account, it is important to understand that question 3, relies on getting the tidy data from the module for question 4, in order to fulfil the expectatins from this section of the project. To aid this, a file **tidy.csv** will be generated in [Question_4.ipynb](https://github.com/etiennedemers/ift6758/blob/master/ift6758-project-template-main/notebooks/Question_4.ipynb)

## Instructions given 

Included in this repo is an image of the NHL ice rink that you can use in your plots.
It has the correct location of lines, faceoff dots, and length/width ratio as the real NHL rink.
Note that the rink is 200 feet long and 85 feet wide, with the goal line 11 feet from the nearest edge of the rink, and the blue line 75 feet from the nearest edge of the rink.

<p align="center">
<img src="./figures/nhl_rink.png" alt="NHL Rink is 200ft x 85ft." width="400"/>
<p>

The image can be found in [`./figures/nhl_rink.png`](./figures/nhl_rink.png).

## Installation

To install this package, first setup your Python environment (next section), and then simply run

    pip install -e .

assuming you are running this from the root directory of this repo.

## Environments

The first thing you should setup is your isolated Python environment.
You can manage your environments through either Conda or pip.
Both ways are valid, just make sure you understand the method you choose for your system.
It's best if everyone on your team agrees on the same method, or you will have to maintain both environment files!
Instructions are provided for both methods.

**Note**: If you are having trouble rendering interactive plotly figures and you're using the pip + virtualenv method, try using Conda instead.

### Conda 

Conda uses the provided `environment.yml` file.
You can ignore `requirements.txt` if you choose this method.
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed on your system.
Once installed, open up your terminal (or Anaconda prompt if you're on Windows).
Install the environment from the specified environment file:

    conda env create --file environment.yml
    conda activate ift6758-conda-env

After you install, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-conda-env

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you make updates to your conda `environment.yml`, you can use the update command to update your existing environment rather than creating a new one:

    conda env update --file environment.yml    

You can create a new environment file using the `create` command:

    conda env export > environment.yml

### Pip + Virtualenv

An alternative to Conda is to use pip and virtualenv to manage your environments.
This may play less nicely with Windows, but works fine on Unix devices.
This method makes use of the `requirements.txt` file; you can disregard the `environment.yml` file if you choose this method.

Ensure you have installed the [virtualenv tool](https://virtualenv.pypa.io/en/latest/installation.html) on your system.
Once installed, create a new virtual environment:

    vitualenv ~/ift6758-venv
    source ~/ift6758-venv/bin/activate

Install the packages from a requirements.txt file:

    pip install -r requirements.txt

As before, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-venv

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you want to create a new `requirements.txt` file, you can use `pip freeze`:

    pip freeze > requirements.txt



