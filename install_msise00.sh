#!/usr/sh

mkdir /tmp/msise00_source
cd /tmp/msise00_source
git clone https://github.com/scivision/msise00 /tmp/msise00_source/msise00
python /tmp/msise00_source/msise00/setup.py install
python -c "import msise00; msise00.build()"
echo "Cleaning build"
rm -Rv /tmp/msise00_source