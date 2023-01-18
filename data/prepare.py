import gdown
import subprocess

url='https://drive.google.com/uc?id=1xv21wHiH9U_f8GfQU9d2MRzT2iDyfg51'
output = 'wikitext2.tgz'

gdown.download(url, output, quiet=False)
subprocess.run("tar -xvzf wikitext2.tgz".split(" "))
subprocess.run("rm wikitext2.tgz".split(" "))
