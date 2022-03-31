import pandas as pd

results = pd.read_csv('results_nooversampling/timm_from-scratch.csv',sep='\t')

for s,foo in iter(results.groupby('epoch')):
    print(s, len(pd.unique(foo['Model'])))

finished_models = set(results[results['epoch'] == 0]['Model'])

bb = pd.read_csv('commands/commands.csv')
bb = bb.sort_values(['input_size','batch_size'], ascending=[True,False])
bb['seed_arg'] = '--seed'
bb['seed'] = 222
bb['pretrained_arg'] = ''
bb['pretrained'] = ''
bb.file = '/cmlscratch/sdooley1/face.evoLVe.PyTorch/fairness_train_timm.py'
bb = bb[bb.batch_size == 100]
bb['batch_size'] = 250
bb = bb.loc[bb.apply(lambda x: x.backbone_name not in finished_models, axis=1)]
bb.to_csv('commands/scratch.sh', sep=' ', index=False, header = False)
