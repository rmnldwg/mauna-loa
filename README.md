# DVC example repository

In this repo you can find a simple but hopefully helpful example on how to get started with [DVC], a **D**ata **V**ersion **C**ontrol software that goes quite a bit beyond just what the name suggests.

[DVC]: https://dvc.org

## Play around with it

To explore how I have used DVC in this repo, just clone it to your computer and `cd` into it.

```
git clone https://github.com/rmnldwg/mauna-loa.git
cd mauna-loa
```

Next, you can install the requirements to execute all the scripts and programs necessary. But first, create a virtual environment for good measure. How you do it is up to you, but I like `venv` a lot:

```
python3 -m venv .venv
source .venv/bin/activate
```

Now your terminal should have a `(.venv)` prefix. Proceed by installing the prerequisites:

```
pip install --upgrade pip
pip install -r requirements.txtx
```

Now you're good to go! Start with the [`dvc repro`][dvc-docs-repro] command to reproduce the entire pipeline.

[dvc-docs-repro]: https://dvc.org/doc/command-reference/repro

## Structure

The structure of this repository is (hopefully) quite self-explanatory:

- `/data` contains the data. Both the raw data, obtained from the [official website][mauna-loa] of the Mauna Loa observatory, as well as the preprocessed and split training & testing data.
- `/scripts` holds python scripts that each perform a specific task. E.g., splitting the data into train & test parts or do the actual inferece, plotting, etc. **git-managed**.
- `/models` stores the trained model settings/parameters. In this case `.npy` arrays.
- `/plots` contains some visualizations of the data and the fit.
- `params.yaml` is a YAML file in which all *hyper*parameters are stored. Decoupled from the scripts like this, it is easier for us and for DVC to keep track of them when we play around with them.
- `metrics.json` is a JSON file where the `eval.py` script stores the performance of the trained model on the test data. 
- `dvc.yaml` describes to DVC what to do with you data, scripts and everything. It contains detailed instructions what to execute with which arguments, what that command depends on and what it produces. That way, DVC can build a directed graph of dependencies and outputs, consequently keeping track of anything that did or did not change while you work on something. **git-managed**


[mauna-loa]: https://gml.noaa.gov/ccgg/trends/data.html

## Branches

The `main` branch in this repository does everything very carefully according to best practices with DVC. But for comparison, there's also the `no-dvc` branch that does things more traditionally only using git. If you'd work with that one, I bet at some point you would face the very problem DVC was invented to solve: You have a broken state in you git history 😨

Lastly there's the `notebook` branch which 'simply' uses a Jupyter notebook to implement the whole pipeline. This is also great and comes very naturally to any data scientist. But as soon as one goes beyong prototyping this can get cluttered and cumbersome. It is also susceptible to the same issues as the vanilla git repo.

## ⚠️ Disclaimer

I am not telling anybody that DVC is the one and only way to do data science. Quite the opposite: I am trying to show you that there are a range of tools out there and each one is designed for a specific purpose in mind. Let me show you what I think:

### Jupyter notebooks are amaying if you

- want to quickly capture a train of thought
- need to prototype something
- would like to present text, math and code in a written document

### Plain old git is amazing if you

- develop 'only' a codebase/package
- work on something incrementally
- want to share your work/codebase with others

### DVC excels at

- allowing you to also version large/huge files
- versioning your pipelines
- comparing different runs of your experiments
- ensuring reproducability
- **(bonus)** forcing you to adopt a number of best-pracices:
    1. modularize your workflow
    2. make your code more resilient
    3. document what your do in a pythonic way (docstrings!)
    4. add outputs (print/log) to your programs/scripts

Some of the above bonuses can easily be perceived to be downsides of using DVC. And if they actually get in your way for a particular project, DVC might not be the right tool for *that* job. But generally, those things only improve your work and your developer skills.