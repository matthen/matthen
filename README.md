```
mkdir ~/virtualenv
python3 -m venv ~/virtualenv/matthen
. ~/virtualenv/matthen/bin/activate
pip install -r requirements.txt
mkdir gh-pages
cd gh-pages
git clone https://github.com/matthen/matthen.git .
git checkout origin/gh-pages -b gh-pages
git branch -d master
cd ..
python build.py
```
