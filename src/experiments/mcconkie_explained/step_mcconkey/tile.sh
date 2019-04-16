#!/usr/bin/env bash

prepath=`echo ~/Research/L2STATS/nnfixrec/`
path="/src/experiments/mcconkie_explained/step_mcconkey/"



#Vertical tile
montage -mode concatenate -tile 1x2 \
$prepath$path"/chance_correct_hb.png" \
$prepath$path"/chance_correct_en.png" \
$prepath/$path"/aux.png"

convert aux.png -bordercolor White -border 2%x0% -gravity east aux.png

convert aux.png -gravity center -pointsize 24  -font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf\
 -draw "rotate 90   text -239,-290 'Hebrew'"  aux.png
convert aux.png -gravity center -pointsize 24  -font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf\
 -draw "rotate 90   text 199,-290 'English'" "tile_step_mcconkey.png"

 rm aux.png



 exit
#Horizontal tile
#add white space
in=$prepath$path"/chance_correct_hb.png"
out=hb_border.png


 convert \
   $in \
   -gravity West \
   -background white \
   -extent $(identify -format '%[fx:W+70]x%H' $in) \
    $out

# -border 4%x0% \ #-extent 590x441 \

# $prepath$path"/chance_correct_hb.png" \
montage -mode concatenate -tile 2x1 \
$prepath$path"/hb_border.png" \
$prepath$path"/chance_correct_en.png" \
$prepath/$path"/aux.png"

convert aux.png -bordercolor White -border 2%x0% -gravity east aux.png

convert aux.png -gravity center -pointsize 20  -font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf\
 -draw "rotate 90   text 0,20 'Hebrew'"  aux.png
convert aux.png -gravity center -pointsize 20  -font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf\
 -draw "rotate 90   text 0,-620 'English'" "htile_step_mcconkey.png"


exit
