import pandas as pd
import numpy as np
import os
import plotly.express as px


def stat_parity_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = g_df[epoch_columns].sum(axis=0)/g_df.shape[0]
    return abs(data['male'] - data['female'])

def stat_parity_ratio_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = g_df[epoch_columns].sum(axis=0)/g_df.shape[0]
    return data['male']/data['female']

def stat_parity_from_rank_ratio_func(df, epoch_columns):
    # calculate the violation of statistical parity
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df[epoch_columns] == 0).sum(axis=0)/g_df.shape[0]
    return data['male']/data['female']

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
    return data['male']/data['female']

def acc_func(df, epoch_columns):
    # calculate the accuracy 
    return df[epoch_columns].sum(axis=0)/df.shape[0]

def acc_from_rank_func(df, epoch_columns):
    # calculate the accuracy 
    return (df[epoch_columns] == 0).sum(axis=0)/df.shape[0]

def get_name_details(f):
    experiment, _ = os.path.splitext(os.path.basename(f))
    experiment = experiment.replace('config_','').replace('_kacc','')
    head = experiment.split('_')[-2]
    opt = experiment.split('_')[-1]
    model = '_'.join(experiment.split('_')[:-2])
    return experiment, model, head, opt


def analyze_files(files, metadata, ratio=False):
    acc_df = pd.DataFrame(columns=['epoch_'+str(e) for e in range(100)])
    acc_disp_df = pd.DataFrame(columns=['epoch_'+str(e) for e in range(100)])
    for f in files:
        df = pd.read_csv(f)
        epoch_columns = df.drop('ids',axis=1).columns
        df = metadata.merge(df)
        num_epochs = len(epoch_columns)
        df[epoch_columns] = df[epoch_columns].apply(lambda x: x == df['label'])
        acc = acc_func(df, epoch_columns)
        experiment = os.path.dirname(f).split('/')[1]
        acc_df.loc[experiment] = acc
        if ratio:
            acc_disp = stat_parity_ratio_func(df, epoch_columns)
        else:
            acc_disp = stat_parity_func(df, epoch_columns)
        acc_disp_df.loc[experiment] = acc_disp    
    return acc_df, acc_disp_df

def analyze_rank_files(files, metadata, ratio=False):
    acc_df = pd.DataFrame(columns=['epoch_'+str(e) for e in range(100)])
    acc_ratio_df = pd.DataFrame(columns=['epoch_'+str(e) for e in range(100)])
    rank_df = pd.DataFrame(columns=['epoch_'+str(e) for e in range(100)])
    for f in files:
        df = pd.read_csv(f)
        epoch_columns = df.drop('ids',axis=1).columns
        df = metadata.merge(df)
        num_epochs = len(epoch_columns)
        acc = acc_from_rank_func(df, epoch_columns)
        experiment = os.path.dirname(f).split('/')[1]
        acc_df.loc[experiment] = acc
        if ratio:
            acc_disp = stat_parity_ratio_func(df, epoch_columns)
        else:
            acc_disp = stat_parity_func(df, epoch_columns)
        acc_ratio_df.loc[experiment] = acc_disp    
        if ratio:
            rank_ratio = rank_ratio_func(df, epoch_columns)
        else:
            rank_ratio = rank_func(df, epoch_columns)
        rank_df.loc[experiment] = rank_ratio    
    return acc_df, acc_ratio_df, rank_df

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