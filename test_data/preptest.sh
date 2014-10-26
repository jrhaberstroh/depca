#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ ! -e $DIR/temp ]; then
    mkdir $DIR/temp
fi
CFG=test.cfg
rm $DIR/temp/*
cp $DIR/$CFG $DIR/temp/$CFG
sed -i "s;#DIR;$DIR;" $DIR/temp/$CFG
ls $DIR/md* > $DIR/temp/csv_files.txt
