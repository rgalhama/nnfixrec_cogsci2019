"""
    Perform two way ANOVA.

"""
__author__ = 'Raquel G. Alhama'

from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols


def anova_mcconkey_test(data):
    """
        2 WAY ANOVA.
        Formula for mcconkey test: correct ~ C(entropy) + C(fixpos) + C(entropy):C(fixpos)
        :param data: pandas dataframe
        :return: p value for entropy, p value for fixation, p value for the interaction, full table
    """

    formula = 'correct ~ C(entropy) + C(fixpos) + C(entropy):C(fixpos)'
    model = ols(formula, data).fit()
    try:
        aov_table = statsmodels.stats.anova.anova_lm(model, typ=2)
    except Exception: #Avoid LinAlg exception for identical performance; return non-significant
        return 1,1,1, None

    #Extract relevant data
    p_entropy = aov_table.loc["C(entropy)","PR(>F)"]
    p_fixation = aov_table.loc["C(fixpos)","PR(>F)"]
    p_inter = aov_table.loc["C(entropy):C(fixpos)","PR(>F)"]

    return p_entropy, p_fixation, p_inter, aov_table


def anova_stepwise(data):
    """
    Performs 2 way ANOVA on the results of the model in the McConkey test.
    It returns the p values only.
    This was only used for debugging purposes.
    Most of the code is taken from the freely shared code in: https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
    :param data: pandas dataframe
    :return:
    """

    #Degrees of freedom
    N = len(data.correct)
    df_a = len(data.entropy.unique()) - 1
    df_b = len(data.fixpos.unique()) - 1
    df_axb = df_a*df_b
    df_w = N - (len(data.fixpos.unique())*len(data.entropy.unique()))

    #Grand mean
    grand_mean = data['correct'].mean()

    #Sum of squares (SS)
    ssq_a = sum([(data[data.entropy == l].correct.mean()-grand_mean)**2 for l in data.entropy])
    ssq_b = sum([(data[data.fixpos == l].correct.mean()-grand_mean)**2 for l in data.fixpos])
    ssq_t = sum((data.correct - grand_mean)**2)
    ##residual (sum of squares within):
    neg = data[data.entropy == 'extreme_neg']
    pos = data[data.entropy == 'extreme_pos']
    neg_fixpos_means = [neg[neg.fixpos == d].correct.mean() for d in neg.fixpos]
    pos_fixpos_means = [pos[pos.fixpos == d].correct.mean() for d in pos.fixpos]
    ssq_w = sum((pos.correct - pos_fixpos_means)**2) +sum((neg.correct - neg_fixpos_means)**2)

    ##SS interaction
    ssq_axb = ssq_t-ssq_a-ssq_b-ssq_w

    #Factor means
    ms_a = ssq_a/df_a
    ms_b = ssq_b/df_b
    ms_axb = ssq_axb/df_axb
    ms_w = ssq_w/df_w

    #F stat
    f_a = ms_a/ms_w
    f_b = ms_b/ms_w
    f_axb = ms_axb/ms_w

    #P-values
    p_a = stats.f.sf(f_a, df_a, df_w)
    p_b = stats.f.sf(f_b, df_b, df_w)
    p_axb = stats.f.sf(f_axb, df_axb, df_w)

    #for the moment, return p-values
    return p_a, p_b, p_axb



