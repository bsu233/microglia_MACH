#!/bin/bash

ls -al *png | awk '{ print $NF }' > list

for i in `cat list`
do
	t=${i:1:3}
	celltype=${t,,}
	id=${i//[!0-9]/}
 	newname="${celltype}${id}.png"
	mv $i $newname
done

