from cmdstanpy import CmdStanModel, CmdStanGQ
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

# from utility import get_data,get_classifier
from sklearn.model_selection import train_test_split

import os
import glob

import argparse

from aif360.datasets import AdultDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric
# import subprocess

import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility import get_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import os
import argparse
import copy
from WAE import data_dis
from numpy import mean, std

from aif360.datasets import BinaryLabelDataset
from sko.PSO import PSO


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'german', 'compas','default', 'bank', 'mep1', 'mep2'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'lr'], help="Classifier name")
parser.add_argument("-p", "--protected", type=str, required=True,
                    help="Protected attribute")

args = parser.parse_args()

dataset_used = args.dataset
attr = args.protected
clf_name = args.clf

# dataset_used = 'compas'
# attr = 'sex'
# clf_name = 'lr'


# def metrics_measure(raw_result, base_result):
#     metric = (1 * (1 - (raw_result[0]/base_result[0]))) + (1 * (1 - (raw_result[3]/base_result[3]))) + ((raw_result[5])/base_result[5]) + (raw_result[6]/base_result[6]) + (raw_result[7]/base_result[7])
#     return metric

def metrics_measure(raw_result):
    metric = (1 * (1 - raw_result[0])) + (1 * (1 - raw_result[3])) + raw_result[5] + raw_result[6] + raw_result[7]
    return metric

def stan_fitting(dataset_orig_train, label, attr):
    # causal model fitting

    fit_data = dataset_orig_train
    fit_data['Probability'] = fit_data['Probability'].astype(int)
    fit_data[attr] = fit_data[attr].astype(int)
    X_rest = fit_data.drop(columns=[label, attr])

    # 2. stan data
    stan_data = {
        'N': len(fit_data),
        'count_rest': fit_data.shape[1] - 2, 
        'Y': fit_data[label].values,
        'X_attr': fit_data[attr].values,
        'X_rest': X_rest.values,
        'N_new': len(fit_data)
    }

    # 3. define stan model

    stan_model_code = """
    data {
    int<lower=0> N; // Number of observations
    int<lower=0> count_rest;
    array[N] int<lower=0, upper=1> Y; // Binary outcome variable
    vector[N] X_attr; // Binary outcome variable
    matrix[N, count_rest] X_rest; // bool features matrix
    int<lower=0> N_new; // Number of new observations to generate
    }
    parameters {
    vector[count_rest] beta_rest; 
    real beta_attr; // Coefficients for predictors
    real alpha; // Intercept
    }
    model {
    beta_rest ~ normal(0, 5); // Weakly informative prior
    beta_attr ~ normal(0, 5); // Weakly informative prior
    alpha ~ normal(0, 10);
    Y ~ bernoulli_logit(alpha + X_rest * beta_rest + X_attr * beta_attr); // Logit model
    }
    generated quantities {
    // int y_pred[N];
    vector[N] y_pred;
    vector[N] y_pred2;
    vector[N] y_pred3;
    vector[N] y_pred4;
    vector[N] y_new;
    vector[N_new] X_attr_new; // New predictor matrix
    X_attr_new = X_attr;

    int i;
    real j;
    real k;

    i = (N!=21341);
    j = 0;
    k = 0.75;

    

    for (n in 1:N) {
        y_pred[n] = bernoulli_logit_rng(alpha + X_rest[n] * beta_rest + (X_attr[n]) * beta_attr);
        y_pred2[n] = bernoulli_logit_rng(alpha + X_rest[n] * beta_rest + (1 - X_attr[n]) * beta_attr);
        y_pred3[n] = bernoulli_logit_rng(alpha + X_rest[n] * beta_rest + (1) * beta_attr);
        y_pred4[n] = bernoulli_logit_rng(alpha + X_rest[n] * beta_rest + (0) * beta_attr);

        y_new[n] = inv_logit(alpha + X_rest[n] * beta_rest + (1-X_attr[n]) * beta_attr);
    }
    }
    """

    # save model
    stan_file = dataset_used + '_' + attr +'.stan'
    with open(stan_file, 'w') as file:
        file.write(stan_model_code)

    # make /Users/showing/Library/CloudStorage/OneDrive-Personal/A-Creating/PhD/Engineering/ICSE24-Multi-Attribute-Fairness-main/Fair360/adult_sex
    # result = subprocess.run(['make', stan_file.replace('.stan', '')], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # kk
    model = CmdStanModel(stan_file=stan_file)
    # model.compile()

    # kk

    # fitting
    fit = model.sample(data=stan_data, iter_sampling=60, iter_warmup=20, chains=1, max_treedepth=15)
    print("Causal Model Fitted!")
    # kk

    return model, fit
# PSO opt
def PSO_func(x):
    global dataset_orig_train
    global dataset_new_train
    global dataset_orig_valid
    global dataset_orig_test
    global df_temp
    global moo_x1
    global moo_x2
    global moo_clf
    global moo_metric
    global p_array_2
    # global attr_index
    # global attr2_index

    # x1, x2 = x  # alpha1, alpha2
    x1 = x  # alpha1, alpha2

    # attr_index = dataset_new_train.feature_names.index(attr)
    df_temp['final_label'] = df_temp['org_label']
    Num1 = min(int(x1 * (len(dataset_orig_train.labels))), len(p_array_2))
    kth_largest1 = np.partition(p_array_2, -Num1)[-Num1]
    p_f = 0.2
    if kth_largest1 <p_f:
      kth_largest1 = p_f
    df_temp.loc[(df_temp['org_attr'] == 0) & (df_temp['new_label_p'] > kth_largest1), 'final_label'] = 1
    dataset_new_train.labels = df_temp['final_label']

    clf = get_classifier(clf_name)
    if clf_name == 'svm':
        clf = CalibratedClassifierCV(base_estimator = clf)
    clf = clf.fit(dataset_new_train.features, dataset_new_train.labels)

    valid_df_copy = copy.deepcopy(dataset_orig_valid)
    pred_valid = clf.predict(dataset_orig_valid.features).reshape(-1,1)
    valid_df_copy.labels = pred_valid

    valid_result1 = measure_final_score(dataset_orig_valid,valid_df_copy, privileged_groups,unprivileged_groups, attr2)
    # metric1 = metrics_measure(valid_result1, base_result)
    metric1 = metrics_measure(valid_result1)
    # print('metric1:', metric1)
    if metric1 < moo_metric:
        moo_metric = metric1
        moo_x1 = x1
        # moo_x2 = x2
        print('moo_metric:', moo_metric)
        # moo_clf = clf
        # moo_x = x
        # print('', moo)
    return metric1


# global dataset_orig_train
# global dataset_orig_test
# global optimal_metrics
# global optimal_round_result
# optimal_round_result = []


label = 'Probability'
scaler = MinMaxScaler()
macro_var = {'adult': ['sex','race'], 'compas': ['sex','race'], 'default':['sex','age'], 'mep1': ['sex','race'],'mep2': ['sex','race']}
df_temp = pd.DataFrame()

multi_attr = macro_var[dataset_used]
for attr_tmp in multi_attr:
    if attr_tmp != attr:
        attr2 = attr_tmp



dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()
privileged_groups = [{attr: 1}]
unprivileged_groups = [{attr: 0}]
privileged_groups2 = [{attr2: 1}]
unprivileged_groups2 = [{attr2: 0}]

results = {}
performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd', 'aod', 'eod', 'spd2', 'aod2', 'eod2']
# performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0-0','spd0-1', 'spd0', 
#                     'aod0-0','aod0-1', 'aod0',  'eod0-0','eod0-1','eod0', 'spd1-0','spd1-1', 'spd1',
#                     'aod1-0','aod1-1', 'aod1', 'eod1-0','eod1-1','eod1', 'wcspd-00','wcspd-01',
#                     'wcspd-10','wcspd-11','wcspd', 'wcaod-00','wcaod-01','wcaod-10','wcaod-11',
#                     'wcaod', 'wceod-00','wceod-01','wceod-10','wceod-11','wceod']
for p_index in performance_index:
    results[p_index] = []

np.random.seed(100)
#split training data and test data
dataset_orig_TV, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
if clf_name == 'rf':
    dataset_orig_train, dataset_orig_valid = train_test_split(dataset_orig_TV, test_size=0.3, shuffle=True)
else:
    dataset_orig_train = copy.deepcopy(dataset_orig_TV)
    dataset_orig_valid = copy.deepcopy(dataset_orig_TV)

scaler.fit(dataset_orig_train)
dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
dataset_orig_valid = pd.DataFrame(scaler.transform(dataset_orig_valid), columns=dataset_orig.columns)
dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)
causal_model, fit_info = stan_fitting(dataset_orig_train, label, attr)

global moo_metric
global base_result
moo_metric = 10.0
repeat_time = 20
pre = 2
x1_set = []
x2_set = []
x_set = []
p_f = 0.2
for r in range(0,repeat_time + pre):
    
    print('===========================================================================================')
    print (r)
    np.random.seed(r)
    #split training data and test data
    dataset_orig_TV, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
    if clf_name == 'rf':
        dataset_orig_train, dataset_orig_valid = train_test_split(dataset_orig_TV, test_size=0.3, shuffle=True)
    else:
        dataset_orig_train = copy.deepcopy(dataset_orig_TV)
        dataset_orig_valid = copy.deepcopy(dataset_orig_TV)
    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_valid = pd.DataFrame(scaler.transform(dataset_orig_valid), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    fit_data = dataset_orig_train
    # print(fit_data.head())
    fit_data['Probability'] = fit_data['Probability'].astype(int)
    fit_data[attr] = fit_data[attr].astype(int)

    X_rest = fit_data.drop(columns=[label, attr])
    # 2. stan data
    stan_data = {
        'N': len(fit_data),
        'count_rest': fit_data.shape[1] - 2,  
        'Y': fit_data[label].values,
        'X_attr': fit_data[attr].values,
        'X_rest': X_rest.values,
        'N_new': len(fit_data)
    }

    generated_quantities = causal_model.generate_quantities(
        data=stan_data,
        previous_fit=fit_info
        # previous_fit=csv_files
        # mcmc_sample=fit
    )


    generated_Y = generated_quantities.y_new

    # mean_generated_attr = np.mean(generated_attr, axis=0)
    mean_generated_Y = np.mean(generated_Y, axis=0)

    df_temp['org_attr'] = dataset_orig_train[attr]
    df_temp['org_label'] = dataset_orig_train['Probability']
    df_temp['new_label'] = mean_generated_Y.round().astype(int)
    df_temp['new_label_p'] = mean_generated_Y
    df_temp['final_label'] = df_temp['org_label']

    not_equal_mask = (df_temp['org_attr'] != 1)
    not_equal_mask_2 = (df_temp['org_attr'] != 1) & (df_temp['org_label'] != 1) 


    p_array = df_temp.loc[not_equal_mask, 'new_label_p'].values
    p_array_2 = df_temp.loc[not_equal_mask_2, 'new_label_p'].values

    # df_temp['org_attr'] = dataset_orig_train[attr]
    # df_temp['org_attr2'] = dataset_orig_train[attr2]
    # df_temp['new_attr'] = mean_generated_Y.round().astype(int)
    # df_temp['new_attr_p'] = mean_generated_Y
    # df_temp['org_label'] = dataset_orig_train[label]
    # df_temp['final_attr'] = df_temp['org_attr']

    # not_equal_mask = (df_temp['org_attr'] != 1)
    # not_equal_mask_2 = (df_temp['org_attr'] != 1) & (df_temp['label'] != 1) 


    # p_array = df_temp.loc[not_equal_mask, 'new_attr_p'].values
    # p_array_2 = df_temp.loc[not_equal_mask_2, 'new_attr_p'].values

    dataset_new_train = copy.deepcopy(dataset_orig_train)
    # dataset_new_train[attr]=1

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                                protected_attribute_names=[attr])
    dataset_orig_valid = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_valid, label_names=['Probability'],
                                protected_attribute_names=[attr])
    dataset_new_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_new_train, label_names=['Probability'],
                                protected_attribute_names=[attr])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                            label_names=['Probability'],
                                            protected_attribute_names=[attr,attr2])
    # print(dataset_new_train.protected_attributes)
    
    clf_base = get_classifier(clf_name)
    if clf_name == 'svm':
        clf_base = CalibratedClassifierCV(base_estimator = clf_base)
    clf_base = clf_base.fit(dataset_orig_train.features, dataset_orig_train.labels)

    base_df_copy = copy.deepcopy(dataset_orig_valid)
    pred_base = clf_base.predict(dataset_orig_valid.features).reshape(-1,1)
    base_df_copy.labels = pred_base
    base_result = measure_final_score(dataset_orig_valid,base_df_copy, privileged_groups,unprivileged_groups, attr2)
    metric_base = metrics_measure(base_result)
    print('metric_base:', metric_base)
    # if metric_base<moo_metric:
    #     moo_metric = metric_base
    if r<pre:
        moo_metric = 10.0
            

        # pso = PSO(func=PSO_func, dim=2, pop=10, max_iter=5, lb=[0.0, 0.0], ub=[0.99, 0.99], w=0.8, c1=0.5, c2=0.5)
        pso = PSO(func=PSO_func, dim=1, pop=10, max_iter=5, lb=[0.0], ub=[0.1], w=0.8, c1=0.5, c2=0.5)
        pso.run()
        x1_set.append(moo_x1)
        # x2_set.append(moo_x2)
        # x_set.append(moo_x)
    else:
        final_x1 = mean(x1_set)
        # final_x2 = mean(x2_set)

        # print(x1_set,x2_set)
        print(x1_set)

        df_temp['final_label'] = df_temp['org_label']
        Num1 = min(int(final_x1 * (len(dataset_orig_train.labels))), len(p_array_2))
        kth_largest1 = np.partition(p_array_2, -Num1)[-Num1]
        df_temp.loc[(df_temp['org_attr'] == 0) & (df_temp['new_label_p'] > kth_largest1), 'final_label'] = 1
        dataset_new_train.labels = df_temp['final_label']
        equal_ratio = (df_temp['org_label'] == df_temp['final_label']).mean()
        print('Equal_ratio:', equal_ratio, final_x1, kth_largest1)



        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator = clf)
        clf = clf.fit(dataset_new_train.features, dataset_new_train.labels)
        

        test_df_copy = copy.deepcopy(dataset_orig_test)
        pred_de = clf.predict(dataset_orig_test.features).reshape(-1,1)
        test_df_copy.labels = pred_de
        round_result= measure_final_score(dataset_orig_test,test_df_copy,privileged_groups,unprivileged_groups, attr2)
        metric_test = metrics_measure(round_result)
        # print("Metric Test", metric_test, final_x1, final_x2)
        print("Metric Test", metric_test, final_x1)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])



val_name = "Fairabel_cy_{}_{}_{}.txt".format(clf_name,dataset_used,attr)
fout = open(val_name, 'w')
for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    # fout.write('\n')
    fout.write('\t%f\t %f\n' % (mean(results[p_index]), std(results[p_index])))
fout.close()
