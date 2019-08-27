#!/bin/bash
cd data/
wget --no-check-certificate -O data.zip https://www.dropbox.com/s/yxfmv9jbig8g7ji/data.zip
unzip data.zip
rm data.zip
cd ..