from setuptools import setup

setup(
    name='PRWKV',
    version='0.2.0',    
    description='This project aims to make RWKV Accessible to everyone using a Hugging Face like OOP interface. Research done by BlinkDL.',
    url='https://github.com/ArEnSc/Production-RWKV',
    author='Michael Chung',
    author_email='michael.chung@databites.ca',
    license='MIT',
    packages=["prwkv"],
    install_requires=['tokenizers',
                      'numpy',
                      'torch',
                      'wget'               
                      ],
    include_package_data = True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Environment :: MacOS X',
        'Environment :: Win32 (MS Windows)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Programming Language :: Python :: 3.9'
    ],
)