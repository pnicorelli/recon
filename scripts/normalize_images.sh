#!/usr/bin/env bash
i=0
IFS=$'\n'

for FILE in  $(find  ../dataset/ -type f -iname '*.*')
do
  i=$(($i+1))
  EXT="${FILE##*.}"
  DIR=$(dirname "${FILE}")
  NAME=$(basename "${FILE}")
  CURFILE="$DIR/$i.$EXT"
  mv $FILE $CURFILE
  convert $CURFILE -resize 100x100\> -channel RGBA $CURFILE
done
