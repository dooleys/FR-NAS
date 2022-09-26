import pandas as pd
import numpy as np
import os
import plotly.express as px
import glob
from collections import Counter


def stat_parity_rank_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df[epoch_columns]==0).sum(axis=0)/g_df.shape[0]
    return abs(data['male'] - data['female'])

def stat_parity_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df[epoch_columns]==0).sum(axis=0)/g_df.shape[0]
    return abs(data['male'] - data['female'])

def stat_parity_ratio_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df[epoch_columns]==0).sum(axis=0)/g_df.shape[0]
    return np.abs(1-data['male']/data['female'])

def stat_parity_ratio_rank_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df[epoch_columns]==0).sum(axis=0)/g_df.shape[0]
    return np.abs(1-data['male']/data['female'])

def ratio_errors_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df[epoch_columns]!=0).sum(axis=0)/g_df.shape[0]
    return np.abs(1-data['male']/data['female'])

def stat_parity_from_rank_ratio_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df[epoch_columns] == 0).sum(axis=0)/g_df.shape[0]
    return np.abs(1-data['male']/data['female'])

def rank_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df[epoch_columns]).sum(axis=0)/g_df.shape[0]
    return abs(data['male']-data['female'])

def rank_ratio_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df[epoch_columns]).sum(axis=0)/g_df.shape[0]
    return np.abs(1-data['male']/data['female'])

def acc_func(df, epoch_columns):
    # calculate the accuracy 
    return df[epoch_columns].sum(axis=0)/df.shape[0]

def acc_from_rank_func(df, epoch_columns):
    # calculate the accuracy 
    return (df[epoch_columns] == 0).sum(axis=0)/df.shape[0]

def err_from_rank_func(df, epoch_columns):
    # calculate the accuracy 
    return (df[epoch_columns] != 0).sum(axis=0)/df.shape[0]

def _resolve_head(s):
    if 'CosFace'.lower() in s.lower():
        return 'CosFace'
    elif 'ArcFace'.lower() in s.lower():
        return 'ArcFace'
    elif 'MagFace'.lower() in s.lower():
        return 'MagFace'
    return 'fault'
    
def _resolve_opt(s):
    if 'AdamW'.lower() in s.lower():
        return 'AdamW'
    if 'Adam'.lower() in s.lower():
        return 'Adam'
    if 'SGD'.lower() in s.lower():
        return 'SGD'
    if 'RMSProp'.lower() in s.lower():
        return 'RMSProp'
    return 'fault'


def get_name_details(f):
    f = f[:-1] if f[-1] == '/' else f
    head_id = -8 if 'cosine' in f else -6
    y = os.path.splitext(os.path.basename(f))[0]
    experiment = y.replace('config_','')
    head = _resolve_head(f)
    opt = _resolve_opt(f)
    head_i = experiment.lower().index(head.lower())
    model = '_'.join(experiment[:head_i].split('_')[:-1])
    return experiment, model, head, opt


def analyze_files(files, metadata, ratio=False, error=False, epochs=None):
    acc_df = pd.DataFrame(columns=['epoch_'+str(e) for e in range(100)])
    acc_disp_df = pd.DataFrame(columns=['epoch_'+str(e) for e in range(100)])
    for f in files:
        try:
            df = pd.read_csv(f)
        except:
            continue
        epoch_columns = df.drop('ids',axis=1).columns
        df = metadata.merge(df)
        num_epochs = len(epoch_columns)
        df[epoch_columns] = df[epoch_columns].apply(lambda x: x == df['label'])
        acc = acc_func(df, epoch_columns)
        experiment = get_name_details(f)[0]
        acc_df.loc[experiment] = acc
        if ratio:
            if error:
                acc_disp = ratio_errors_func(df, epoch_columns)
            else:
                acc_disp = stat_parity_ratio_func(df, epoch_columns)
        else:
            acc_disp = stat_parity_func(df, epoch_columns)
        acc_disp_df.loc[experiment] = acc_disp    
    return acc_df, acc_disp_df

def analyze_rank_files_np(files, metadata, ratio=False, error=False, epochs=None): 
    if epochs is None:
        epochs = ['epoch_'+str(e) for e in range(100)]
        
    acc_df = pd.DataFrame(columns=epochs)
    acc_ratio_df = pd.DataFrame(columns=epochs)
    rank_df = pd.DataFrame(columns=epochs)
    
    for f in files:
        try:
            df = pd.read_csv(f)
        except:
            continue
        epoch_columns = list(set(df.columns).intersection(epochs))
        df = metadata.merge(df)
        num_epochs = len(epoch_columns)
        if error:
            acc = err_from_rank_func(df, epoch_columns)
        else:
            acc = acc_from_rank_func(df, epoch_columns)
        experiment = get_name_details(f)[0]
        acc_df.loc[experiment] = acc
        
        if ratio:
            if error:
                acc_disp = ratio_errors_func(df, epoch_columns)
            else:
                acc_disp = stat_parity_ratio_rank_func(df, epoch_columns)
        else:
            acc_disp = stat_parity_rank_func(df, epoch_columns)
        acc_ratio_df.loc[experiment] = acc_disp 
        
        if ratio:
            rank_ratio = rank_ratio_func(df, epoch_columns)
        else:
            rank_ratio = rank_func(df, epoch_columns)
        rank_df.loc[experiment] = rank_ratio    
    return acc_df, acc_ratio_df, rank_df

def analyze_rank_files(files, metadata, ratio=False, error=False, epochs=None): 
    acc_df, acc_ratio_df, rank_df = analyze_rank_files_np(files, metadata, ratio=ratio, error=error, epochs=epochs)
    return prepare(acc_df), prepare(acc_ratio_df), prepare(rank_df)

def plot_df(acc_df, acc_disp_df, rank_df = None, title = ''):
    def prepare(df):
        # dataframe of a long format
        df = pd.melt(df.reset_index(), id_vars='index')
        df = df.rename(columns={'variable':'epoch'})
        df = df.rename(columns={'value':'Accuracy'})
        df.epoch = df.epoch.apply(lambda x: int(x.split('_')[1]))
        return df
    acc_df = prepare(acc_df)
    acc_disp_df = prepare(acc_disp_df)

    # plotly express
    acc_df['measurement'] = 'Accuracy'
    acc_disp_df['measurement'] = 'Disparity'

    df = acc_df.append(acc_disp_df)

    if rank_df is not None:
        rank_df = prepare(rank_df)
        rank_df['measurement'] = 'Rank'
        df = df.append(rank_df)
        
    df = df.dropna()

    fig = px.line(df, x='epoch', y='Accuracy', color='index', facet_row='measurement', title=title)
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(yaxis_title="Disparity")

    fig.show()
    
    
def whatIsPareto(df, x_inc=False, y_inc=False):
    isPareto = np.zeros(df.shape[0])
    i = 0
    for _, (c1,c2) in df.iterrows():
        tmp = np.delete(np.array(df), (i), axis=0)
        if x_inc: # is a larger x better?
            if y_inc: # is a larger y better?
                b = np.any(np.apply_along_axis(lambda x: x[0]>c1 and x[1]>c2, 1, tmp))
            else: # is a smaller y better?
                b = np.any(np.apply_along_axis(lambda x: x[0]>c1 and x[1]<c2, 1, tmp))
        else: # is a smaller x better?
            if y_inc: # is a larger y better?
                b = np.any(np.apply_along_axis(lambda x: x[0]<c1 and x[1]>c2, 1, tmp))
            else: # is a smaller y better?
                b = np.any(np.apply_along_axis(lambda x: x[0]<c1 and x[1]<c2, 1, tmp))
        if not b:
            isPareto[i] = 1
        i+=1
    return isPareto

def preparePareto(df, x_inc=False, y_inc=False):
    
    isPareto = whatIsPareto(df, x_inc=x_inc, y_inc=y_inc)
    tmp = df[isPareto == 1]
    
    tmp = tmp.sort_values(df.columns[0])
    return tmp

def prepare(df):
    # dataframe of a long format
    df = pd.melt(df.reset_index(), id_vars='index')
    df = df.rename(columns={'variable':'epoch'})
    df = df.rename(columns={'value':'Metric'})
    df.epoch = df.epoch.apply(lambda x: int(x.split('_')[1]))
    return df

def merge(df1, df2):
    df = df1.merge(df2, on=["index","epoch"])
    df = df.rename(columns={'Metric_x':'Accuracy'})
    df = df.rename(columns={'Metric_y':'Disparity'})
    return df

def drop_models(df_list, models):
    # remove rows with model names in models from each df in the df_list
    out_list = []
    for df in df_list:
        out_list += [df[~df['index'].isin(models)]]
    return out_list


def find_yaml_folder(yaml):
    '''
    given a yaml string file like:
         'config_inception_resnet_v2_CosFace_RMSProp.yaml'
    return the corresponding folder for this experiment:
         './Phase1B/inception_resnet_v2_CosFace_RMSProp'
    if it does not exist, return ''
    '''
    experiment_name = yaml.replace('config_','').replace('.yaml','')
    
    R2_or_Phase1B = 'R2' if 'R2' in yaml else 'Phase1B'
            

    experiment_folders = glob.glob('/cmlscratch/sdooley1/merge_timm/FR-NAS/Checkpoints/{}/*/'.format(R2_or_Phase1B))
    if R2_or_Phase1B == 'Phase1B':
        experiment_folders += glob.glob('/cmlscratch/sdooley1/merge_timm/FR-NAS/Checkpoints/timm_explore_few_epochs/*/')
    where = [get_name_details(experiment_name)[0].lower() == get_name_details(x)[0].lower() for x in experiment_folders]
    yaml_folder = ''
    if any(where):
        yaml_folder = experiment_folders[np.where(where)[0][0]]
    return yaml_folder

def get_finished_models_Phase1B():
    '''
    Return a list of those models which we are including in Phase1B
    '''
    finished = []
    for yaml_orig in glob.glob('/cmlscratch/sdooley1/merge_timm/FR-NAS/configs/**/*.yaml') + glob.glob('/cmlscratch/sdooley1/merge_timm/FR-NAS/configs_multi/**/*.yaml'):
        yaml = os.path.basename(yaml_orig)
        yaml_folder = find_yaml_folder(yaml)
        if yaml_folder:
            finished += [yaml]
    cn = Counter([get_name_details(x)[1] for x in finished])
    final_models = [k for k,v in cn.items() if v>=6]
    final_models.sort()
    if 'vit_large_patch16_224' in final_models:
        final_models.remove('vit_large_patch16_224')
    if 'cait_xs24_384' in final_models:
        final_models.remove('cait_xs24_384')
        
    # make sure vgg_bn goes before vgg
    a, b = final_models.index('vgg19'), final_models.index('vgg19_bn')
    final_models[b], final_models[a] = final_models[a], final_models[b]


    return final_models


def get_pareto_hps_head_opt(stable_df, col='Accuracy'):
    row = []
    for opt in ['adamw', 'sgd']:
        for head in ['ArcFace','CosFace','MagFace']:
            df = stable_df
            df = df[(df['opt'] == opt) & (df['head'] == head)]
            ind = whatIsPareto(df[[col,'Disparity']], True, False).astype(bool)
            out = df[ind].dropna().sort_values(col, ascending=False)
            m = out['model'].to_string(header=False,index=False).split('\n')
            row += ['\n'.join(list(np.unique([x.strip() for x in m])))]
    return row

def get_pareto_hps_opt(stable_df, col='Accuracy'):
    row = []
    for opt in ['adamw', 'sgd']:
            df = stable_df
            df = df[(df['opt'] == opt)]
            ind = whatIsPareto(df[[col,'Disparity']], True, False).astype(bool)
            out = df[ind].dropna().sort_values(col, ascending=False)
            m = out['model'].to_string(header=False,index=False).split('\n')
            row += ['\n'.join(list(np.unique([x.strip() for x in m])))]
    return row

def get_pareto_hps_head(stable_df, col='Accuracy'):
    row = []
    for head in ['ArcFace', 'CosFace', 'MagFace']:
            df = stable_df
            df = df[(df['head'] == head)]
            ind = whatIsPareto(df[[col,'Disparity']], True, False).astype(bool)
            out = df[ind].dropna().sort_values(col, ascending=False)
            m = out['model'].to_string(header=False,index=False).split('\n')
            row += ['\n'.join(list(np.unique([x.strip() for x in m])))]
    return row

def anova_hp_accuracy(df, col = 'Accuracy'):
    df['model'] = df['index'].apply(lambda x: get_name_details(x.replace('_rank_by_id_val',''))[1])
    df['head'] = df['index'].apply(lambda x: get_name_details(x.replace('_rank_by_id_val',''))[2])
    df['opt'] = df['index'].apply(lambda x: get_name_details(x.replace('_rank_by_id_val',''))[3].lower())
    df = df.merge(meta, left_on='model', right_on='model_name')
    df.fillna('0',inplace=True)
    df[col] = df[col].astype(float)

    lm = ols(col+' ~ head + opt', data=df).fit() # fitting the model
    
    print(sm.stats.anova_lm(lm))
    tukey_head = pairwise_tukeyhsd(endog=df[col],
                              groups=df['head'],
                              alpha=0.05)
    print(tukey_head)
    tukey_opt = pairwise_tukeyhsd(endog=df[col],
                              groups=df['opt'],
                              alpha=0.05)
    print(tukey_opt)
    
    return sm.stats.anova_lm(lm), tukey_head, tukey_opt

def anova_hp_disp(df, col = 'Accuracy'):
    df['model'] = df['index'].apply(lambda x: get_name_details(x.replace('_rank_by_id_val',''))[1])
    df['head'] = df['index'].apply(lambda x: get_name_details(x.replace('_rank_by_id_val',''))[2])
    df['opt'] = df['index'].apply(lambda x: get_name_details(x.replace('_rank_by_id_val',''))[3].lower())
    df = df.merge(meta, left_on='model', right_on='model_name')
    df.fillna('0',inplace=True)
    df['Disparity'] = df['Disparity'].astype(float)

    lm = ols('Disparity ~ head + opt', data=df).fit() # fitting the model
    
    print(sm.stats.anova_lm(lm))
    tukey_head = pairwise_tukeyhsd(endog=df['Disparity'],
                              groups=df['head'],
                              alpha=0.05)
    print(tukey_head)
    tukey_opt = pairwise_tukeyhsd(endog=df['Disparity'],
                              groups=df['opt'],
                              alpha=0.05)
    print(tukey_opt)
    
    return sm.stats.anova_lm(lm), tukey_head, tukey_opt
