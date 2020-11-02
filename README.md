## Setup

```bash
pyenv virtualenv 3.7.7 matthen
pyenv activate matthen
pip install -r requirements.txt
mkdir gh-pages
cd gh-pages
git clone https://github.com/matthen/matthen.git .
git checkout origin/gh-pages -b gh-pages
git branch -d master
cd ..
```

## Build
```
python build.py
```

## Preview
```
cd gh-pages
python -m http.server
```
