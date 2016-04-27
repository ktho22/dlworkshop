config_str = """
c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 9999
"""

import os
home_dir = os.environ['HOME']

with open(home_dir+"/.jupyter/jupyter_notebook_config.py", "w") as cf:
	cf.write(config_str)
