#!/usr/bin/env python
import os

os.system("jupyter notebook --generate-config")

if os.path.isdir("$HOME/.jupyter/profile_nbserver")is False:
    os.system("ipython profile create nbserver")
else: 
    os.system("echo profile_nbserver is already exist.")

yes = set(['yes','y', 'ye', ''])
no = set(['no','n'])

config_str = """
# Server config
c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
"""

home_dir = os.environ['HOME']

with open(home_dir+"/.jupyter/jupyter_notebook_config.py", "w") as cf:
	cf.write(config_str)
