#!/usr/bin/env python

import glob
import os
import importlib.resources

"""
Generate the code documentation, using pydoc.
"""

# El CSS no funcionava i buscant vaig trobar: https://stackoverflow.com/questions/79272347/how-do-i-format-pydoc-html-output
# On, adaptant el codi, el css per fi ha funcionat
css_data = importlib.resources.files('pydoc_data').joinpath('_pydoc.css').read_text()

for dirname in ['generardataset', 'clustersciclistes']:
    flist = glob.glob('%s/*.py' % dirname)
    for fname in flist:
        if '__init__' not in fname:
            print(fname)
            os.system('python -m pydoc -w %s' % fname)
            bname = os.path.splitext(os.path.basename(fname))[0]
            os.system('mv %s.html %s.%s.html' % (bname, dirname, bname))
    os.system('python -m pydoc -w %s' % dirname)
    
    # Per utilitzar el CSS
    with open(dirname + ".html") as inp:
        html = inp.read()
    with open(dirname + ".html", "w") as out:
        out.write(html.replace("</head>", "<style>%s</style></head>" % css_data))
os.system('mv *.html docs')