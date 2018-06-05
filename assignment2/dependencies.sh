#!/bin/sh
apt-get update  # To get the latest package lists
apt-get install python3
apt-get -y install python3-pip

pip3 install pandas
pip3 install statsmodels
pip3 install sklearn
pip3 install matplotlib
pip3 install nltk