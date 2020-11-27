# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:25:28 2020

@author: 79344
"""

import re
import sys

from sphinx.quickstart import generate, do_prompt
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    d= {
        'path': '.',
        'sep': True,
        'dot': '_',
        'author': 'Lucas M. Uriarte and Luc Lababarriere',
        'version': '0.0.1',
        'release': '0.0.1',
        'suffix': '.rst',
        'master': 'index',
        'epub': False,
        'ext_autodoc': True
        'ext_doctest': False,
        'ext_intersphinx': False,
        'ext_todo': False,
        'ext_coverage': False,
        'ext_pngmath': False,
        'ext_mathiax': False,
        'ext_ifconfig': True,
        'ext_todo': True,
        'ext_viewcode': False,
        'makefile': True,
        'batchfile': False,
    }
    if 'project' not in d:
        print '''
            Nom du projet
        '''
        do_prompt(d, 'project', 'Project Name')
    generate(d) 