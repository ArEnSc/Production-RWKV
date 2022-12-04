from setuptools import setup

setup(
    name='PRWKV',
    version='0.0.1',    
    description='This project aims to make RWKV Accessible to everyone using a Hugging Face like OOP interface. Research done by BlinkDL.',
    url='https://github.com/shuds13/pyexample',
    author='Michael Chung',
    author_email='michael.chung@databites.ca',
    license='MIT',
    packages=["prwkv"],
    install_requires=['tokenizers',
                      'numpy',
                      'torch'                  
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)