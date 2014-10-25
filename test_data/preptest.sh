#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cp $DIR/test_config.cfg $DIR/temp/test_config.cfg
sed -i "s;#DIR;$DIR;" $DIR/temp/test_config.cfg
ls $DIR/md* > $DIR/temp/csv_files.txt
