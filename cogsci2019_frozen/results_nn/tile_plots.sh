#!/usr/bin/env bash


#Create tiles of plots (used for cogsci2019 tile p[lot)


prepath=`echo ~/Research/L2STATS/nnfixrec/results/`

dimming=""
dimming="_dimming0.35"


#Reciepe for cogsci panel figure:
#upper row is hebrew, lower is english
#from uniform to behavioral
#legend: only hb behavioral
#xlabel: only english blend
#hb: title for scheme
#y label: add it with imagemagick (otherwise plots move)
#add language vertical title with imagemagick (here)

### Plot 3x2
montage -mode concatenate -tile 3x2 \
$prepath/hb_uniform20$dimming/means_2runs_epoch200.png \
$prepath/hb_blend50$dimming/means_2runs_epoch200.png \
$prepath/hb_behavioral$dimming/means_2runs_epoch200.png \
$prepath/en_uniform20$dimming/means_2runs_epoch200.png \
$prepath/en_blend50$dimming/means_2runs_epoch200.png \
$prepath/en_behavioral$dimming/means_2runs_epoch200.png \
tile_test$dimming.png


#add border
convert tile_test$dimming.png -bordercolor White -border 2%x0% -gravity south aux.png

convert aux.png -gravity West -pointsize 24  -font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf\
 -draw "rotate -90   text -90,15 'Mean correct'"  aux.png

convert aux.png -gravity center -pointsize 24  -font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf\
 -draw "rotate 90   text -150,-490 'Hebrew'"  aux.png
convert aux.png -gravity center -pointsize 24  -font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf\
 -draw "rotate 90   text 150,-490 'English'" tile_test$dimming.png
#add vertical text
#convert aux.png -gravity West -pointsize 20 -font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf \
# -annotate +0+0 "$(echo -n "Mean correct" | sed 's/./&@/g; s/@$//' | tr '@' '\012')" aux.png

#convert aux.png -gravity East -pointsize 24 -font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf -annotate +-1+0 "$(echo -n "Hebrew       English" | sed 's/./&@/g; s/@$//' | tr '@' '\012')" aux.png

rm aux.png
exit
### Plot 2x3
montage -mode concatenate -tile 2x3 \
$prepath/hb_behavioral$dimming/means_2runs_epoch200.png \
$prepath/hb_blend50$dimming/means_2runs_epoch200.png \
$prepath/hb_uniform20$dimming/means_2runs_epoch200.png \
$prepath/en_behavioral$dimming/means_2runs_epoch200.png \
$prepath/en_blend50$dimming/means_2runs_epoch200.png \
$prepath/en_uniform20$dimming/means_2runs_epoch200.png \
tile_test$dimming.png

#add border
convert tile_test$dimming.png -bordercolor White -border 0%x1% -gravity south aux.png

#annotate
convert aux.png -gravity South -pointsize 25 -annotate -310+4 'Hebrew' aux.png
convert aux.png -gravity South -pointsize 25 -annotate +320+4 'English' "tile_test"$dimming"_annotated.png"
rm aux.png