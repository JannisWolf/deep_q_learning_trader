FAU 2019 Preparations
===

## Abstract
Python application to show AI functionality based on Keras and TensorFlow.
This is used for teaching at FAU 2019.

## Table Of Contents
* [Abstract](#abstract)
* [Table Of Contents](#table-of-contents)
* [Overview](#overview)
* [Components](#components)
  * [Stock Exchange](#stock-exchange) 
  * [Trader](#trader)
  * [Predictor](#predictor)
* [Required Tools](#required-tools)
  * [Installing Python and pip 3 on Mac](#installing-python-3-and-pip-on-mac)
  * [Installing Python and pip 3 on Windows](#installing-python-3-and-pip-on-windows)
  * [Optional: Installing virtualenv](#optional-installing-virtualenv)
* [Run the Application](#run-the-application)
  * [Clone the Repository](#clone-the-repository)
  * [Create a Virtual Environment (optional)](#create-a-virtual-environment-optional-)
  * [Install All Dependencies](#install-all-dependencies)
  * [Run](#run)
* [Development](#development)
  * [IDE](#ide)
  * [Overview Of This Repository](#overview-of-this-repository)
* [Authors](#authors)

## Overview
This Python application simulates a computer-based stock trading program.
Its goal is to demonstrate the basic functionality of neural networks trained by reinforcement learning
(deep Q-learning).

The application consists of a stock exchange and several connected traders.
The stock exchange asks each trader once per day for its orders, and executes any received ones.
Each trader computes its orders based on
(1) stock market information provided by the stock exchange, and
(2) stock votes provided by some experts.
Both information may be inputs to the trader's neural network trained by reinforcement learning.

The following resources provide some basic introductions into the topic of neural networks and reinforcement learning:
* AI
  * [The AI Revolution: The Road to Superintelligence](https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-1.html)
* Neural networks
  * [A Brief Introduction to Neural Networks](http://www.dkriesel.com/science/neural_networks)
* Deep reinforcement learning
  * [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
  * [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
  * [Demystifying Deep Reinforcement Learning - Intel Nervana](https://www.intelnervana.com/demystifying-deep-reinforcement-learning/)
  * [30 Amazing Applications of Deep Learning](http://www.yaronhadad.com/deep-learning-most-amazing-applications/)
  
* Python
  * [Learn Python - Free Interactive Python Tutorial](https://www.learnpython.org/en/Welcome)

## Components
### Stock Exchange
The stock exchange represents the central 'metronome' of the application.
It is implemented by a class 'StockExchange'.
The stock exchange maintains both the stock prices and the trader's portfolios.
This means that all traders connected to the stock exchange are assigned one portfolio which the stock exchange manages
to prevent fraud.
A portfolios comprises not only the inventory of all stocks and their quantity, but also the available cash amount. 

The stock exchange emulates trading days by calling the connected traders.
To keep it simple the traders are only called once at the end of the day.
The stock exchange then provides each trader with both the latest close prices and its respective portfolio.
A trader is supposed to reply with a list of orders which will be executed during the next day.
An order is one of the following actions for all stocks that are traded at the stock exchange: Buy or sell.
After obtaining all orders for all connected traders the stock exchange executes the orders one by one. 
This is only limited by checks whether the specific order is valid for a given portfolio.
That means, for buying stocks the portfolio's cash reserve must suffice.
For selling stocks, the corresponding quantity of stocks must reside in the portfolio.
Cash gained from stock sales will only be available for stock purchases the following day.

After executing all orders for all connected traders the current trading day has ended and the next one begins.
  
### Trader
Each trader is implemented by a separate trader class (e.g., 'BuyAndHoldTrader' or 'DeepQLearningTrader').
A trader gets the latest close prices and its current portfolio, and returns a list of orders to the stock exchange.
For computing the orders, a trader may employ a previously trained neural network.
Most traders additionally rely on one or more stock experts in the background.

### Expert
Each expert is implemented by a separate expert class (e.g., 'ObscureExpert').
A expert works behind a trader and provides a vote (buy, hold, or sell) for a specific stock. 

## Required Tools
This application relies on Python 3, thus the following tools are required:
* Python 3
* pip (may come with your Python installation)
* virtualenv (optional)

Details on how to install these tools are listed below.

### Installing Python 3 and pip on Mac
On Mac there are two ways to install Python 3:
* The installer way: Visit https://www.python.org/downloads/release/python-363/ to install Python 3
* The Homebrew way: Visit http://docs.python-guide.org/en/latest/starting/install3/osx/ for a tutorial to install 
Python 3 using Homebrew

Check if *pip* is installed with running `$ pip --version`. In case it is not already installed:
* When using the installer: Install *pip* separately by running `$ python get-pip.py` after downloading 
[get-pip.py](https://bootstrap.pypa.io/get-pip.py)
* When using Homebrew: Execute `$ brew install pip`

### Installing Python 3 and pip on Windows
A good tutorial can be found here: http://docs.python-guide.org/en/latest/starting/install3/win/.
To ease running Python in the Command Line you should consider adding the Python installation directory to the
PATH environment variable.

Check if *pip* is installed with running `$ pip --version`. In case it is not already installed run 
`$ python get-pip.py` after downloading [get-pip.py](https://bootstrap.pypa.io/get-pip.py).

### Optional: Installing virtualenv
The easiest and cleanest way to install all required dependencies is *virtualenv*. This keeps all dependencies in a 
specific directory which in turn will not interfere with your system's configuration. This also allows for easier 
version switching and shipping.

To install *virtualenv* run `$ pip install virtualenv`

## Run the Application
After installing all required tools (Python, *pip*, *\[virtualenv]*) execute the following commands:

### Clone the Repository
```
$ git clone <repository url>
$ cd <repository folder>
```

### Create a Virtual Environment (optional)
If you want to use *virtualenv*, create a virtual environment. The directory *virtual_env* is already added to 
*.gitignore*.

#### On Mac
```
$ virtualenv -p python3 virtual_env
$ source virtual_env/bin/activate
```

#### On Windows
```
$ virtualenv -p [path\to\python3\installation\dir\]python virtual_env
$ virtual_env/Scripts/activate
```

### Install All Dependencies
This installs all required dependencies by Trader.AI.
```
$ pip install -r requirements.txt
```

### Run
```
$ python stock_exchange.py
```
After some Terminal action this should show a diagram depicting the course of different portfolios which use different
Trader implementations respectively.

Furthermore you can execute the test suite to see if all works well:
```
$ python test_runner.py
```

## Development
### IDE
There are no specific requirements for developing a Python application.
You can open your favorite text editor (notepad.exe, TextEdit, vim, Notepad++, sublime, Atom, emacs, ...),
type in some code and run it with `$ python your-file.py`.
However, there are some IDEs which make developing and running Python applications more convenient.
We worked with the following:
* [JetBrains PyCharm](jetbrains.com/pycharm/)
* [PyDev](http://www.pydev.org/) (based on Eclipse)

In your IDE you may have to select the correct Python environment.
Most IDEs can detect the correct environment automatically.
To check and - if needed - select the correct Python installation directory or the *virtual_env* 
directory inside your repository do as follows:
* **PyCharm**: Visit "Preferences" > "Project: xxx" > "Project Interpreter" and check if the correct environment 
is selected. If not, select the gear symbol in the upper right
* **PyDev**: Visit "Window" > "Preferences" > "PyDev" > "Interpreters" > "Python Interpreter" and check if the correct
environment is selected. If not, select "New..."

### Overview Of This Repository
This repository contains a number of packages and files.
The following provides a short overview:
* datasets - CSV dumps of the used stock prices
* experts - Python package that contains all experts
* framework - Python package that contains the whole framework around the stock exchange
* traders - Python package that contains all traders
* `directories.py` - Contains some project-wide Python constants
* `README.md` - This file
* `requirements.txt` - Contains an export of all project dependencies (by running `$ pip freeze > requirements.txt`)
* `stock_exchange.py` - Contains the central main method. This starts ILSE
* `test_runner.py` - Runs the test suite with all test cases

## Authors
* [Richard Müller](mailto:richard.mueller@senacor.com)
* [Christian Neuhaus](mailto:christian.neuhaus@senacor.com)
* [Jonas Holtkamp](mailto:jonas.holtkamp@senacor.com)
* [Janusz Tymoszuk](mailto:janusz.tymoszuk@senacor.com)