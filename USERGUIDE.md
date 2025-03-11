
# User Guide: Setting Up `config.ini`

This guide explains how to set up your `config.ini` file for the **Daanish** project.
At the end, save the config.ini file in the root directory of your Daanish project.

----------------------------------------------

## Create a `config.ini` File
Create a `config.ini` file in the root of the project with the following content:


[global]
PYTHONPATH = /path/to/your/project

Replace /path/to/your/project with the absolute path to your Daanish project folder.

Example for Windows:
[global]
PYTHONPATH = C:\Data Science Projects\Daanish

Example for macOS/Linux:
[global]
PYTHONPATH = /Users/your-username/Data Science Projects/Daanish

----------

[DATABASE]
server = a\SQLEXPRESS
database = ML
username = Admin
password = 123
use_database = False  

Replace your_database_server, your_database_name, your_username, and your_password with your database credentials.
Set use_database to True if you want to use a database, or False to use CSV files.


----------------------------------------------------------------------
# Template `config.ini` file

I have prepared a template config.ini file 'config.ini.template'. Remove the .template extension and edit the file as needed.

