rm -rf ./build
rm -rf ./dist
python setup.py check
python setup.py sdist
pip3 install .