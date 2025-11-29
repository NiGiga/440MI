# Tutorial: GitHub Actions Step by Step

## What is GitHub Actions?
GitHub Actions is a CI/CD platform integrated into GitHub that allows you to automate workflows such as build, test, and deploy directly from your repository.

## 1. Prerequisites
- An active account on Github
- A repository on GitHub
- Write permissions on the repository

## 2. Basic Structure
GitHub provides preconfigured workflow templates that you can use as-is or customize to create your own workflow. This templates are in format .yaml

GitHub analyzes your code and shows you workflow templates that might be useful for your repository. For example, if your repository contains Node.js code, you'll see suggestions for Node.js projects. 


## 3. How to

The steps to follow in order to integrate a github actions workflow are:

- Navigate to the desired repository;
- On the upper bar, there are many options (code, issues, pull requests, actions, projects etc.). Select "Actions" from the bar;
- A dashboard with various template options will appear;
- From this dashboard, you can easily select one template and configure it as preferred.

GitHub Actions workflows are defined in YAML files and must be inside the `.github/workflows` folder. Within the `.github/workflows` folder, it's possible to keep more then one .yaml, if necessary.

If instead you desire to create it on your own from scratch, just create a folder `.github/workflows` within the project folder, create the .yaml and then push to the project repository,


## Example

For each push that occurs, Github must run the command `python manage.py test`

```yaml
name: Test Django

on:
  push:
  schedule:
    - cron: "0 2 * * *"

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python manage.py test


```

- The `name` keyword is optional.
- The `on: push` defines the trigger, that would be the event that communicates to the workflow when it needs to be activated. Basically, it get activated whenever the action push is performed by the user. 
This test runs on a virtual machines, that is provided by github.

- `Cron` explanation:

```
0 2 * * *
↑ ↑ ↑ ↑ ↑
│ │ │ │ └── giorno della settimana (qualsiasi)
│ │ │ └──── mese (qualsiasi)
│ │ └────── giorno del mese (qualsiasi)
│ └──────── ora = 2
└────────── minuto = 0
```

In this case, the cron is setup to start everyday at 2.00, indipendently from the day, or the month (as they are not defined).

With the following set of commands, we are defining two different trigger events. 

```yaml
on:
  push:
  schedule:
    - cron: "0 2 * * *"

```

Basically, the workflow will be activated into two scenarios:

- When a push to the repository occurs;
- Everyday at 2.00am the workflow is activated automatically.


At the steps sections:

- The `jobs` keyword groups together all the jobs that run in the workflow.  
- The `test` keyword is the name of the job
- `runs-on` start a virtual machine with ubuntu distribution
- `uses`: actions/checkouts@v4 &rarr; takes care of cloning the repository whitin the virtual machine.
- `uses`: actions/setup-python@v4 &rarr; this action is installing Python.
- `with`: defines the Python version to install.
- The `run` keyword tells the job to execute a specific command on the runner. In this case, we are installing the packages needed in order to run the next command (that needs django library).


For the entire metadata syntax, refer to the [docs](https://docs.github.com/en/actions/reference/workflows-and-actions/metadata-syntax).


## 6. Monitoring
After pushing, go to the "Actions" tab in your GitHub repository to see the status of your workflows.

## 7. Tips
- Use secrets for sensitive variables
- You can create multiple workflows for different purposes
- Customize trigger events


## 8. Debugging
If a workflow fails, click on the job to see detailed logs and fix any errors.


## 9. Advanced Example: Matrix Build

```yaml
name: Test Matrix
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node: [16, 18]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}
      - run: npm ci
      - run: npm test
```

In this workflow, we are testing two different version of Node.js. These two versions are defined in the `strategy: matrix: node: [16,18]` section. We are telling "create two identical execution of the job for two different versions of Node.js version 16 and Node.js version 18". GitHub Actions will automatically duplicate the job. This is an useful way to parallelize the same task for more then one version.

It's possible to parallelize even the virtual machines on which to run the test. For example, you might want to define a workflow that tests Node.js both on an vm with OS ubuntu and windows.

## Example:

name: Test Matrix
on: [push]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]   # OS for the vm
        node: [16, 18]                       # Node.js versions
    runs-on: ${{ matrix.os }}                # the runner uses matrix OS

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}   # Node versions from the matrix

      - run: npm ci
      - run: npm test


## 7. Useful Resources
- [Official Documentation](https://docs.github.com/en/actions)
- [Marketplace Actions](https://github.com/marketplace?type=actions)

