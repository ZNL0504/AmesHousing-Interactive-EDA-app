# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 19:20:46 2023

@author: ZNL
"""
import pandas as pd
import numpy as np
from scipy import stats # import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols # for ANOVA
from statsmodels.stats.proportion import proportions_ztest, proportions_chisquare
import researchpy as rp # for t-test and ANOVA
import dash_bootstrap_components as dbc
# from tabulate import tabulate # for table printing
# from sklearn.preprocessing import OrdinalEncoder
# import pathlib
# import json

# # =============================================================================
# # Load dataset, data wrangling, pre-defined lists/dicts
# # =============================================================================
# PATH = pathlib.Path(__file__).parent
# DATA_PATH = PATH.joinpath('data').resolve()

# ames = pd.read_csv(DATA_PATH.joinpath('ames_new.csv'))
# ames = ames[['Sale_Price'] + [col for col in ames.columns if col != 'Sale_Price']]

# with open(DATA_PATH.joinpath('var_type_dict.json'), 'r') as f:
#     var_type_dict = json.load(f)

# var_types = list(var_type_dict.keys()) # list of variable types

# twolev_nom_vars = [feat for feat in var_type_dict['nominal'] if len(ames[feat].unique()) == 2]
# twolev_ord_vars = [feat for feat in var_type_dict['ordinal'] if len(ames[feat].unique()) == 2]
# twolev_cat_vars = twolev_nom_vars + twolev_ord_vars


# =============================================================================
# correlation coefficient calculation and interpretation:
# pearson: continuous vs. continuous (both vars are continuous)
# spearman: continuous vs. discrete
# =============================================================================
def corr_coef_interp(df, type_1, type_2, var_1, var_2, method):
    corr_coef = df[var_1].corr(df[var_2], method=method).round(3)
    if abs(corr_coef) < 0.1:
        corr_lev = 'ALMOST NO'
    elif 0.1 <= abs(corr_coef) < 0.4:
        corr_lev = 'A WEAK'
    elif 0.4 <= abs(corr_coef) < 0.6:
        corr_lev = 'A MODERATE'
    else: # abs(corr_coef) >= 0.6: 
        corr_lev = 'A STRONG'
    
    corr_typ = 'LINEAR' if method == 'pearson' else 'MONOTONIC'
    corr_direct=''
    if abs(corr_coef) >= 0.1: 
        corr_direct = ', POSITIVE' if corr_coef > 0 else ', NEGATIVE'
        
    text_1 = ('''##### **Correlation Coefficient Analysis:**\n- **{} variable: `{}`**\n- **{} variable: `{}`**\n'''
              .format(type_1, var_1, type_2, var_2))
    text_2 = '-' * 30 + '\n\n'
    text_3 = '{} correlation coefficient: {}\n\n'.format(method, corr_coef)
    text_4 = ('There is **{} {}{}** correlation between `{}` and `{}`.\n\n'
              .format(corr_lev, corr_typ, corr_direct, var_1, var_2))
    text_interp = text_1 + text_2 + text_3 + text_4
    return text_interp

# =============================================================================
# independent t-test result and interpretation:
# 2-level ordinal & nominal vs. continuous variables
# =============================================================================
def ind_ttest_interp(df, cat_type, num_type, cat_var, num_var, sig_level=0.05):
    cat_lst = list(df[cat_var].unique())
    summary, results = rp.ttest(group1 = df[df[cat_var] == cat_lst[0]][num_var], 
                                group1_name = cat_lst[0],
                                group2 = df[df[cat_var] == cat_lst[1]][num_var], 
                                group2_name = cat_lst[1])
    t_stat, p_value = results['results'][2], results['results'][3]
    
    text_1 = ('##### **Independent t-test:**\n- **{} variable: `{}`**\n- **{} variable: `{}`**\n'
              .format(cat_type, cat_var, num_type, num_var))
    text_2 = '-' * 30 + '\n'
    text_3 = ('(t-statistic, p_value) = ({:.3f}, {:.3f}), significance level = {}\n\n'
              .format(t_stat, p_value, sig_level))
    if p_value < sig_level:
        text_4 = ('''There **EXISTS** a statistically significant difference in 
                    the average `{}` between two groups of `{}`.\n\n'''
                  .format(num_var, cat_var))
        if results['results'][4] < sig_level:
            text_4 += ('''###### The average `{}` in group `{}` is statistically significantly **higher** than that in group `{}`.\n'''
                       .format(num_var, cat_lst[1], cat_lst[0]))

        if results['results'][5] < sig_level:
            text_4 += ('''###### The average `{}` in group `{}` is statistically significantly **higher** than that in group `{}`.\n'''
                       .format(num_var, cat_lst[0], cat_lst[1]))
    else: # p_value >= sig_level
        text_4 = ('''There is **NO** statistically significant difference in the 
                    average `{}` between two groups of `{}`.\n'''
                  .format(num_var, cat_var))
    text_5 = '=' * 50 + '\n'
    # sum_stats_tbl = tabulate(summary.round(4), headers=summary.columns, 
    #                          tablefmt='simple_grid')
    # sum_stats_tbl = summary.round(4).to_markdown(tablefmt='simple_grid')
    sum_stats_tbl = dbc.Table.from_dataframe(summary.round(4), striped=True, 
                                             bordered=True, hover=True,
                                             style={'font-size': '14px'})
    # test_result_tbl = tabulate(results.round(4), headers=results.columns)
    test_result_tbl = dbc.Table.from_dataframe(results.round(4), striped=True,
                                               bordered=True, hover=True,
                                               style={'font-size': '14px'})
    # text_interp = text_1 + text_2 + text_3 + text_4 + text_5 + sum_stats_tbl + text_6 + test_result_tbl
    text_interp = text_1 + text_2 + text_3 + text_4 + text_5
    return text_interp, sum_stats_tbl, test_result_tbl

# =============================================================================
# ANOVA result and interpretation:
# multi-level (> 2) ordinal & nominal vs. continuous variables
# =============================================================================
def anova_interp(df, cat_type, num_type, cat_var, num_var, sig_level=0.05):
    model_str = '{} ~ C({})'.format(num_var, cat_var)
    model = ols(model_str, data = df).fit()
    aov_table = sm.stats.anova_lm(model, typ = 2) # test result table
    # get adjusted R-squared value from ols model summary
    ols_sum = model.summary()
    adj_rsquared = float(ols_sum.tables[0].data[1][3])
    # get summary stats table:
    df_overall = rp.summary_cont(df[num_var])
    df_groups = rp.summary_cont(df.groupby(cat_var)[num_var])
    summary = pd.concat([df_groups, pd.DataFrame(df_overall.iloc[0, 1:].rename('Overall')).T])
    
    f_statistic, p_value = aov_table.F[0], aov_table.iloc[0, -1]
    
    text_1 = ('##### **ANOVA:**\n- **{} variable: `{}`**\n- **{} variable: `{}`**\n\n'
              .format(cat_type, cat_var, num_type, num_var))
    text_2 = '-' * 30 + '\n'
    text_3 = ('(F-statistic, p_value) = ({:.3f}, {:.3f}), significance level = {}\n'
              .format(f_statistic, p_value, sig_level))
    text_3_1 = '\nOLS Regression Model: `{} ~ C({})`\n'.format(num_var, cat_var)
    text_3_2 = '\nAdjusted R-squared value of OLS result: {}\n\n'.format(adj_rsquared)
    if p_value < sig_level:
        text_4 = ('''There **EXISTS** a statistically significant difference in 
                  the average `{}` between groups of `{}`: '''
                 .format(num_var, cat_var))
        text_4 += ('''Among all groups of `{}`, at least one pair of `{}` means are different from each other.\n\n'''
                   .format(cat_var, num_var))
    else: # p_value >= sig_level
        text_4 = ('There is **NO** statistically significant difference in the average `{}` between groups of `{}`.\n\n'
                  .format(num_var, cat_var))
    text_5 = '###### According to R-squared value of OLS Model, about {}% of the variability observed in `{}` can be explained by the regression model.\n'.format(adj_rsquared * 100, num_var)
    text_6 = '=' * 50 + '\n'
    
    sum_stats_tbl = dbc.Table.from_dataframe(summary.round(4), striped=True, 
                                             bordered=True, hover=True,
                                             index=True,
                                             style={'font-size': '14px'})
    test_result_tbl = dbc.Table.from_dataframe(aov_table.round(4), striped=True,
                                               bordered=True, hover=True,
                                               index=True,
                                               style={'font-size': '14px'})
    text_interp = text_1 + text_2 + text_3 + text_3_1 + text_3_2 + text_4 + text_5 + text_6
    return text_interp, sum_stats_tbl, test_result_tbl

# =============================================================================
# two independent proportion z-test/ Fisher's Exact test:
# 2-level ordinal & nominal vs 2-level ordinal & nominal
# =============================================================================
def cat2lev_ind_test_interp(df, type_1, type_2, var_1, var_2, sig_level=0.05):
    crosstab, fisher_test_res, exp_count_tab = rp.crosstab(df[var_1], df[var_2], 
                                                           test='fisher', 
                                                           expected_freqs = True)
    z_test_res = proportions_ztest(count=crosstab.iloc[:2, 0], 
                                   nobs=crosstab.iloc[:2, 2], alternative='two-sided')
    chi2_test_res = proportions_chisquare(count=crosstab.iloc[:2, 0], 
                                          nobs=crosstab.iloc[:2, 2])[:2]
    z_stat, p_val_z = z_test_res
    chi2_stat, p_val_chi2 = chi2_test_res
    p_val_fisher = fisher_test_res.iloc[1, 1]
    res_df= pd.DataFrame({'test_statistic': [z_stat, chi2_stat, np.nan], 
                          'p_value': [p_val_z, p_val_chi2, p_val_fisher], 
                          'cal_function': ['proportions_ztest()', 'proportions_chisquare()', 'crosstab()'],
                          'module': ['statsmodels.stats.proportion', 'statsmodels.stats.proportion', 'researchpy']},
                         index = ['z-test', 'Chi-square test', 'Fisher’s Exact test'])
    
    text_1 = ('##### **Independence Test -- 2-level Categorical Variables:**\n- **{} variable: `{}`**\n- **{} variable: `{}`**\n\n'
              .format(type_1, var_1, type_2, var_2))
    text_2 = '-' * 30 + '\n'
    text_3 = '''**z-test assumption check:**\n'''
    blank_1 = 'NOT SATISFIED' if np.any(crosstab.values < 10) else 'SATISFIED'
    blank_2 = 'NOT ALL' if np.any(crosstab.values < 10) else 'ALL'
    text_3 += ('''According to crosstab (contingency table in graph and summary statistics below), the sample size condition for two independent proportion z-test is **{}**:\n'''
               .format(blank_1))
    text_3 += '''**{}** cells have at least 10 cases.\n\n'''.format(blank_2)
    if p_val_z < sig_level:
        text_4 = ('According to p_value from z-test result (check hypothesis test result table below), There **EXISTS** a relationship between `{}` and `{}` distribution with significance level {}.\n\n'
                  .format(var_1, var_2, sig_level))
        text_4 += '###### The two categorical variables are **DEPENDENT** of each other.\n'
    else: # p_val_z >= sig_level
        text_4 = ('According to p_value from z-test result, There is **NO** relationship between `{}` and `{}` distribution with significance level {}.\n\n'
                  .format(var_1, var_2, sig_level))
        text_4 += '###### The two categorical variables are **INDEPENDENT** of each other.\n'
    text_5 = '=' * 50 + '\n'
    text_interp = text_1 + text_2 + text_3 + text_4 + text_5
    
    sum_stats_tbl = dbc.Table.from_dataframe(crosstab.round(4), striped=True, 
                                             bordered=True, hover=True,
                                             index=True,
                                             style={'font-size': '14px'})
    test_result_tbl = dbc.Table.from_dataframe(res_df.round(4), striped=True,
                                               bordered=True, hover=True,
                                               index=True,
                                               style={'font-size': '14px'})
    return text_interp, sum_stats_tbl, test_result_tbl

# =============================================================================
# Chi-square test of independence:
# between multi-level (> 2) ordinal & nominal + discrete variables
# note: here discrete variables are treated as ordinal categorical variables
# =============================================================================
def chi2_test_interp(df, type_1, type_2, var_1, var_2, sig_level=0.05):
    crosstab_pd = pd.crosstab(df[var_1], df[var_2])
    res_stats = stats.chi2_contingency(crosstab_pd)
    d_freedom = res_stats[2]
    
    crosstab, test_res, exp_count_tab = rp.crosstab(df[var_1], df[var_2], 
                                                    test = "chi-square", 
                                                    expected_freqs = True)
    chi2_statistic, p_val = test_res.iloc[0, 1], test_res.iloc[1, 1]
    
    text_1 = ('##### **Chi-square test--multi-level categorical variables:**\n- **{} variable: `{}`**\n- **{} variable: `{}`**\n\n'
              .format(type_1, var_1, type_2, var_2))
    text_2 = '-' * 30 + '\n'
    text_3 = ('(Chi-square statistic, p_value, degree_freedom) = ({:.3f}, {:.3f}, {}),\nsignificance level = {}\n\n'
              .format(chi2_statistic, p_val, d_freedom, sig_level))
    text_4 = '**Chi-square test assumption check:**\n'
    blank_1 = 'NOT SATISFIED' if np.any(exp_count_tab.values < 5) else 'SATISFIED'
    blank_2 = 'NOT ALL' if np.any(exp_count_tab.values < 5) else 'ALL'
    text_4 += ('According to **expected cell counts** table (contingency table in summary statistics below), the sample size condition for Chi-square test is **{}**:\n'
               .format(blank_1))
    text_4 += ('**{}** cells have at least 5 **expected** cases.\n\n'.format(blank_2))
    if p_val < sig_level:
        text_5 = ('According to p_value from test result, There **EXISTS** a relationship between `{}` and `{}` distribution with significance level {}.\n\n'
                  . format(var_1, var_2, sig_level))
        text_5 += '###### The two variables are **DEPENDENT** of each other.\n'
    else:
        text_5 = ('According to p_value from test result, There is **NO** relationship between `{}` and `{}` distribution with significance level {}.\n\n'
                  . format(var_1, var_2, sig_level))
        text_5 += '###### The two variables are **INDEPENDENT** of each other.\n'
    text_6 = '=' * 40 + '\n'
    text_interp = text_1 + text_2 + text_3 + text_4 + text_5 + text_6
    sum_stats_tbl = dbc.Table.from_dataframe(exp_count_tab.round(4), striped=True, 
                                             bordered=True, hover=True,
                                             index=True,
                                             style={'font-size': '14px'})
    test_result_tbl = dbc.Table.from_dataframe(test_res.round(4), striped=True,
                                               bordered=True, hover=True,
                                               index=True,
                                               style={'font-size': '14px'})
    return text_interp, sum_stats_tbl, test_result_tbl


# =============================================================================
# function that conducts statistical inference:
# call one of the above functions under specific condition
# =============================================================================
def stats_inference(df, twolev_cat_vars_lst, var_type_1, var_type_2, 
                    var_name_1, var_name_2):
    text_interp = 'abc'
    sum_stats_tbl, test_result_tbl = '', ''
    ### correlation coefficient analysis: continuous & discrete
    if (var_type_1 in ['continuous', 'discrete']) and (var_type_2 in ['continuous', 'discrete']):
        method = ('pearson' if (var_type_1 == 'continuous' and var_type_2 == 'continuous') 
                  else 'spearman')
        text_interp = corr_coef_interp(df=df, type_1=var_type_1, 
                                        type_2=var_type_2, 
                                        var_1=var_name_1, var_2=var_name_2,
                                        method=method)
    ### categorical vs. continuous variables
    if (var_type_1 == 'continuous' and var_type_2 in ['ordinal', 'nominal']) or (var_type_2 == 'continuous' and var_type_1 in ['ordinal', 'nominal']):
        cat_type = var_type_1 if var_type_2 == 'continuous' else var_type_2 
        cat_var = var_name_1 if var_type_2 == 'continuous' else var_name_2
        num_var = var_name_1 if var_type_1 == 'continuous' else var_name_2
        # independent t-test: 2-level ordinal & nominal vs. continuous variables
        if (cat_var in twolev_cat_vars_lst):
            text_interp, sum_stats_tbl, test_result_tbl = ind_ttest_interp(df=df, 
                                                                           cat_type=cat_type, 
                                                                           num_type='continuous', 
                                                                           cat_var=cat_var, 
                                                                           num_var=num_var, 
                                                                           sig_level=0.05) 
        # ANOVA: multi-level ordinal & nominal vs continuous variables
        else:
            text_interp, sum_stats_tbl, test_result_tbl = anova_interp(df=df, 
                                                                       cat_type=cat_type, 
                                                                       num_type='continuous', 
                                                                       cat_var=cat_var, 
                                                                       num_var=num_var, 
                                                                       sig_level=0.05)
    ### categorical/discrete vs. categorical/discrete variables
    # independence test between 2-level categorical variables
    if (var_type_1 in ['discrete', 'ordinal', 'nominal']) and (var_type_2 in ['discrete', 'ordinal', 'nominal']):
        if (var_name_1 in twolev_cat_vars_lst) and (var_name_2 in twolev_cat_vars_lst):
            text_interp, sum_stats_tbl, test_result_tbl = cat2lev_ind_test_interp(df=df, 
                                                                                  type_1=var_type_1, 
                                                                                  type_2=var_type_2, 
                                                                                  var_1=var_name_1, 
                                                                                  var_2=var_name_2)
        # chi-squared test between multi-level (> 2) ordinal & nominal + discrete variables
        else:
            text_interp, sum_stats_tbl, test_result_tbl = chi2_test_interp(df=df, 
                                                                           type_1=var_type_1, 
                                                                           type_2=var_type_2, 
                                                                           var_1=var_name_1, 
                                                                           var_2=var_name_2)
        
    return text_interp, sum_stats_tbl, test_result_tbl









