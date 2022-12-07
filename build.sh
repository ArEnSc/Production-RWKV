rm -rf ./build
rm -rf ./dist
rm -rf ./PRWKV.egg-info
python setup.py check
python setup.py sdist
pip3 install .