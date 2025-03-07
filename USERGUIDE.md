
# User Guide: Setting Up `config.ini`

This guide explains how to set up your `config.ini` file for the **Daanish** project.

---

## Create a `config.ini` File
Create a `config.ini` file in the root of the project with the following content:

```ini
[global]
PYTHONPATH = /path/to/your/project

Replace /path/to/your/project with the absolute path to your Daanish project folder.

Example for Windows:
[global]
PYTHONPATH = C:\Data Science Projects\Daanish

Example for macOS/Linux:
[global]
PYTHONPATH = /Users/your-username/Data Science Projects/Daanish


## What is PYTHONPATH?
The PYTHONPATH setting specifies the root directory of your project. This helps Python locate modules and packages within the Daanish project.