__author__ = 'Raquel G. Alhama'

#Path to store temporary results
TMP = "../../../tmp/"


#Probabilities won't sum up to 1 due to floating point arithmetic; this is the allowed deviation \
# when checking for probabilities.
TOLERANCE_PROB = 0.001

#Languages
LANGS = ("hb", "es", "en")
langstr = dict(zip(LANGS, ("Hebrew", "Spanish", "English")))

#Language-specific styles
COLOR_PALETTE={"hb": "#7570b3", "en":"#1b9e77", "es": "#d95f02"}
MARKERS={"hb":'D', "en":"o", "es":">"}

#Palette for experimental design (2x2)
condition_palette = {
    "f2neg": "salmon",
    "f2pos": "indianred",
    "f6neg": "turquoise",
    "f6pos": "teal"
}



