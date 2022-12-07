from setuptools import setup

setup(
    name='PRWKV',
    version='0.0.1',    
    description='This project aims to make RWKV Accessible to everyone using a Hugging Face like OOP interface. Research done by BlinkDL.',
    url='https://github.com/ArEnSc/Production-RWKV',
    author='Michael Chung',
    author_email='michael.chung@databites.ca',
    license='MIT',
    packages=["prwkv"],
    install_requires=['tokenizers',
                      'numpy',
                      'torch'                  
                      ],
    include_package_data = True,
    classifiers=[
        'Development Status :: 1 - Prototype',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux :: OSX :: Windows',        
        'Programming Language :: Python :: 3.9.6',
    ],
)