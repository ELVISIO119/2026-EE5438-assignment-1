"""Build and execute a portable notebook with code, evidence and selected weights."""
import base64
import io
import json
import textwrap
import zipfile
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def build():
    from select_final import write_pareto
    write_pareto()
    recipe=json.loads((OUT/'final_recipe.json').read_text())
    metrics=json.loads((OUT/'final_test_metrics.json').read_text())
    evidence={p.stem:json.loads(p.read_text()) for p in OUT.glob('*.json')}
    class_metrics={name:row for name,row in metrics['classification_report'].items() if isinstance(row,dict) and 'avg' not in name}
    weakest=min(class_metrics,key=lambda name:class_metrics[name]['f1-score'])
    classes=list(class_metrics)
    confusions=sorted([(count,classes[i],classes[j]) for i,row in enumerate(metrics['confusion_matrix']) for j,count in enumerate(row) if i!=j],reverse=True)[:3]
    files=[ROOT/name for name in ('models.py','train.py','baseline.py','sharpness.py','prune.py',
                                  'evaluation.py','select_final.py','evaluate_final.py','garment_experiment.py',
                                  'cascade_experiment.py','benchmark_cascade.py','benchmark_nvfp4.py',
                                  'image_processing_experiment.py','benchmark_awq.py',
                                  'run_yolo.py','select_yolo.py','check_pyramid.py',
                                  'pil_experiment.py','feature_experiment.py','moe_experiment.py','loss_followup.py',
                                  'hybrid_experiment.py','refine_cascade.py','latent_search.py','import_legacy.py','SOURCES.md','requirements.txt')]
    files+=list((ROOT/'configs').glob('*.json'))+list(OUT.glob('*.json'))+list(OUT.glob('*.csv'))
    files += [OUT/f'{name}.pt' for name in sorted({'baseline_sgd_sigmoid','legacy_clean','legacy_muon'}|{c['name'] for c in recipe['components']})]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.relative_to(ROOT))
    payload=base64.b64encode(buffer.getvalue()).decode('ascii')
    cells=[]
    md=lambda source:cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))
    code=lambda source:cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))
    md(f'''# Assignment 1: Image Classification with Multi-Layer Perceptrons

    **Cai Haochen | Student ID 58561440 | EE5438 | Semester A 2026-2027**

    **Section 17's automated configuration search reaches {evidence['latent_test_metrics']['selected']['balanced']['accuracy']:.2%} test accuracy and 301.33 million average test MACs**, with unchanged model storage. Its 4,096 cached candidates took 10.37 seconds to screen, followed by real inference and timing of five finalists. The selected recipe was committed before test evaluation. This searches a compact eleven-dimensional configuration space, not learned image features.

    The preceding **adaptive-view classifier in Section 16** obtains **{evidence['adaptive_test_metrics']['selected']['balanced']['accuracy']:.2%} test accuracy**, using 339.42 million average test MACs and 3,164,836 stored parameters. It adds fine-patch views only when the initial fallback ensemble remains uncertain, reusing completed work. Default Run All evaluates it and preceding frozen options. The class-conditional reference in **Section 15** uses 410.16 million average test MACs at the same observed accuracy; Section 14 uses 475.71 million. Section 13 preserves earlier hybrids, including the smaller-storage accuracy-priority option.

    The frozen reference classifier, retained in Sections 1–12, obtains **{metrics['accuracy']:.2%} test accuracy**, macro precision **{metrics['macro_precision']:.4f}**, macro recall **{metrics['macro_recall']:.4f}**, and macro F1 **{metrics['macro_f1']:.4f}** on 10,000 Fashion-MNIST test images. All components are MLPs. No convolution, attention, Transformer, outside training images or externally pretrained weights are used. The new front-stage weights come from this student's earlier training on the same assignment split; their provenance is retained.

    A confidence gate first evaluates one unflipped view with the wide Mixer. Predictions of Trouser, Sandal, Sneaker, Bag or Ankle boot with probability at least 0.90 exit immediately. All other images use the selected ensemble detailed below. The same wide-model weights are shared between stages; no extra model is stored. The gate uses predictions only, never the true class.

    The technical summary and limitations at the end explain which changes helped, which did not, and both average and worst-case inference cost. Accuracy meets the assignment's highest published accuracy threshold when at least 93%; marks and top-ten bonus are determined by the instructor, not guaranteed by this notebook.
    ''')
    md('''## 1. Reproducibility and data discipline

    **Run All performs real evaluation of the frozen weights.** The portable bundle below contains readable Python sources, configuration files, recorded training histories, and the selected checkpoints. It is unpacked into a new temporary working directory, leaving existing files untouched. The bundle is not a remote dependency.

    Set `RETRAIN_ALL=True` to rerun the repository's training sequence from scratch before validation selection and test evaluation. This excludes the two imported legacy checkpoints: their original configurations, histories and hashes are retained, but the optional training sequence does not recreate their original training. Sections 13–17 evaluate frozen hybrid endpoints only in default mode; after retraining, their stored hash bindings no longer apply and a new validation-only freeze is required. Training is opt-in because it includes all negative-result experiments. Recorded histories are not represented as newly executed training. PyTorch 2.10, torchvision 0.25, NumPy, pandas, scikit-learn and matplotlib are required. CUDA is recommended; CPU evaluation is supported but slower.

    The official 60,000-image training set is stratified into 54,000 training and 6,000 validation examples, 600 per validation class, using seed **58561440**. Mean and standard deviation come only from the 54,000 training images. Validation chooses checkpoints, averaging, pruning, inference views and ensemble weights. The test recipe is frozen and checkpoint hashes are checked before its evaluation. The public test benchmark has been evaluated during development; this continuation is not an independently blinded test. New candidate fitting and selection use training/validation data only. Re-evaluating the frozen recipe checks implementation consistency, not an independent replication of generalization.
    ''')
    code(f'''import base64, io, os, sys, tempfile, zipfile
from pathlib import Path
RETRAIN_ALL = False
runtime = Path(tempfile.mkdtemp(prefix='ee5438_assignment1_'))
bundle = base64.b64decode({payload!r})
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
    archive.extractall(runtime)
os.chdir(runtime)
sys.path.insert(0, str(runtime))
print('Self-contained workspace:', runtime)
''')
    cells[-1].metadata={'jupyter':{'source_hidden':True},'tags':['bundle']}
    code('''import json, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision.datasets import FashionMNIST
from IPython.display import display
print({'Python':platform.python_version(), 'PyTorch':torch.__version__,
       'torchvision':torchvision.__version__, 'CUDA':torch.cuda.is_available()})
''')
    md('''## 2. Architecture and implementation

    The reference MLP is `784 -> 128 -> 64 -> 10`, with sigmoid hidden activations and SGD (learning rate 0.001, batch size 128, 50 epochs). Its low score reflects an undertrained fixed reference recipe, not an optimized SGD ceiling. Cross-entropy consumes raw logits; softmax is applied when probabilities are needed.

    Residual variants use a 784-to-256 input projection, three pre-normalized residual blocks, a final LayerNorm and a ten-class head. Blocks compare GELU, SiLU and approximately parameter-matched SwiGLU. BatchNorm replaces block LayerNorm in its explicit control, while the final LayerNorm remains.

    The dense Mixer reshapes a 28x28 image into 49 non-overlapping 4x4 patches, embeds each with a Linear layer, and alternates token and channel MLPs. Six blocks use width 128; the student uses width 64 and three blocks. Token mixing is shared across channels and channel mixing across tokens. LayerNorm, residual additions, GELU and mean pooling complete this MLP-only design. There is no convolutional patch embedding or attention.

    The accuracy-first extension also trains a 2x2-patch Mixer (196 tokens, width 96, six blocks), a larger 4x4-patch Mixer (width 192, eight blocks), and a flat-input residual SwiGLU MLP (width 768, four blocks). The Muon-trained Mixer receives a separate augmented refinement trial. These are complete recipe comparisons, not isolated architectural causal effects. All architectures remain pure MLPs.

    A subsequent readout experiment replaces mean pooling with flattening followed by a linear classifier. Repeating the original head weights across token positions and dividing by the number of tokens reproduces the original mean-pooled logits in real arithmetic at initialization. This tests positional discrimination with added parameters; it does not introduce convolution or attention. These trials are recorded even when they overfit and are not selected.
    ''')
    code('''from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(14,3))
labels=['28 x 28 image\\nNormalize', 'Reshape patches\\nLinear embedding',
        'Residual Mixer blocks\\nToken MLP + channel MLP',
        'LayerNorm + mean\\nLinear head + softmax',
        'Average views\\nWeighted model average']
for i,label in enumerate(labels):
    x=3*i
    ax.add_patch(FancyBboxPatch((x,.5),2.6,1.2,boxstyle='round,pad=0.08',facecolor='#e5eef8',edgecolor='#24486b'))
    ax.text(x+1.3,1.1,label,ha='center',va='center',fontsize=9)
    if i<4: ax.annotate('',xy=(x+2.9,1.1),xytext=(x+2.65,1.1),arrowprops={'arrowstyle':'->'})
ax.set(xlim=(-.2,14.9),ylim=(0,2.2)); ax.axis('off')
ax.set_title('Full ensemble path for images not accepted by the confidence gate')
plt.tight_layout(); plt.show()
''')
    code((ROOT/'models.py').read_text())
    md('''### Training code

    The following cell defines the executable training implementation. Five-epoch linear warmup precedes cosine learning-rate decay to 1% of the peak. AdamW excludes biases and normalization gains from decay. Muon receives only two-dimensional hidden-block matrices; the embedding, classifier, bias and normalization parameters remain on AdamW. Muon uses momentum 0.95, five Newton-Schulz steps and RMS-matched update scaling.

    Geometry uses horizontal flips, rotations within 8 degrees, scale variation within 8%, and translations within two pixels. Raw-image padding is zero. Mixup uses alpha 0.2 and soft targets. Training accuracy under augmentation/Mixup is not directly comparable to clean validation accuracy. Mixer training uses optional CUDA BF16 autocast; parameter storage and optimizer states remain FP32. SAM/ASAM use two gradient evaluations per update; their dropout masks are independent. ASAM uses weight scale `abs(weight)+0.01` for matrices.
    ''')
    code((ROOT/'train.py').read_text().split("if __name__=='__main__':")[0])
    code((ROOT/'sharpness.py').read_text().split("if __name__=='__main__':")[0])
    md('''### Optional full training

    The student distillation loss is `(1-alpha)*CE + alpha*T^2*KL(teacher || student)`, with alpha 0.5 and temperature 3. The teacher is the independently trained, frozen AdamW Mixer; only training images are used for teacher targets. Teacher construction preserves the student's random-number stream. Its paired control uses the same student initialization and training configuration. Teacher pretraining and forward passes are additional training costs, even though student inference needs only the student.

    EMA decay is 0.995 after each minibatch. SWA averages epoch snapshots in the final 20% of one fine-tuning trajectory. Both use native PyTorch averaging and LayerNorm, so no BatchNorm recalibration is needed. Structured pruning removes channel-MLP neurons ranked by the product of incoming and outgoing weight norms. It removes rows/biases from the first matrix and matching columns from the second matrix, reducing physical dimensions; it is not masked dense inference.
    ''')
    code('''if RETRAIN_ALL:
    import runpy
    runpy.run_path('baseline.py', run_name='__main__')
    from train import run
    sequence = ['adamw','residual_gelu','residual_silu','residual_bn','geometry','mixup',
                'swiglu','muon','sam','asam','mixer','mixer_muon','averages',
                'student_control','student_distill']
    for name in sequence:
        run(json.loads(Path(f'configs/{name}.json').read_text()))
    from prune import trials
    trials()
    for name in ['regularization_none','regularization_dropout','regularization_decay']:
        run(json.loads(Path(f'configs/{name}.json').read_text()))
    for name in ['accuracy_p2','accuracy_wide','accuracy_muon_refine','accuracy_residual','accuracy_p2_clean','accuracy_wide_clean']:
        run(json.loads(Path(f'configs/{name}.json').read_text()))
    from select_final import select
    select(accuracy_first=True,exclude_prefixes=('garment_','focal_','spatial_','yolo_','pil_','features_','moe_','loss_','legacy_'))
    from garment_experiment import run as select_refinements
    for name in ['garment_wide','garment_p2','focal_wide','focal_p2']:
        run(json.loads(Path(f'configs/{name}.json').read_text()))
    select_refinements()
    for name in ['spatial_wide','spatial_p2','spatial_pruned','spatial_pruned_garment']:
        run(json.loads(Path(f'configs/{name}.json').read_text()))
    select_refinements(spatial=True)
    from cascade_experiment import run as select_cascade
    select_cascade(Path('results/spatial_recipe.json'))
    Path('results/final_recipe.json').write_text(Path('results/cascade_recipe.json').read_text())
    from run_yolo import run as train_yolo, STEMS
    train_yolo(force=True)
    Path('results/yolo_reference.json').write_text(Path('results/final_recipe.json').read_text())
    from select_yolo import run as select_yolo
    select_yolo(STEMS)
    Path('results/final_recipe.json').write_text(Path('results/yolo_recipe.json').read_text())
    from pil_experiment import run as fit_pil
    fit_pil()
    Path('results/final_recipe.json').write_text(Path('results/pil_recipe.json').read_text())
    from feature_experiment import run as fit_features
    Path('results/features_reference.json').write_text(Path('results/final_recipe.json').read_text())
    fit_features(force=True)
    Path('results/final_recipe.json').write_text(Path('results/features_recipe.json').read_text())
    from moe_experiment import run as fit_moe
    Path('results/moe_reference.json').write_text(Path('results/final_recipe.json').read_text())
    fit_moe(force=True)
    Path('results/final_recipe.json').write_text(Path('results/moe_recipe.json').read_text())
    from loss_followup import run as fit_losses
    Path('results/loss_reference.json').write_text(Path('results/final_recipe.json').read_text())
    fit_losses(force=True)
    Path('results/final_recipe.json').write_text(Path('results/loss_recipe.json').read_text())
else:
    print('Using recorded training evidence and frozen checkpoints; full retraining is disabled.')
''')
    md('## 3. Recorded experiments and validation analysis')
    code('''records = {p.stem:json.loads(p.read_text()) for p in Path('results').glob('*.json')}
rows=[]
for name,r in records.items():
    if 'val_accuracy' not in r: continue
    cfg=r.get('config',{})
    rows.append({'experiment':name,'val_accuracy_%':100*r['val_accuracy'],
                 'parameters':r.get('parameters',109386),'dense_MACs':r.get('macs',109184),
                 'epochs':0 if 'pil_parent' in cfg else cfg.get('epochs',50),'best_epoch':r.get('best_epoch'),
                 'seconds':r.get('seconds'),'averaging':r.get('averaging',''),
                 'shared_training_run':r.get('shared_training_run','')})
table=pd.DataFrame(rows).sort_values('val_accuracy_%',ascending=False)
display(table.round({'val_accuracy_%':3,'seconds':2}))
''')
    code('''fig,axes=plt.subplots(1,2,figsize=(13,4))
for name in ['baseline_sgd_sigmoid','adamw_sigmoid','residual_muon','mixer_adamw','mixer_muon']:
    history=records[name]['history']
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss'] for r in history],label=name)
    axes[1].plot(epochs,[100*r['val_accuracy'] for r in history],label=name)
axes[0].set(xlabel='Epoch',ylabel='Training objective',title='Recorded training curves')
axes[1].set(xlabel='Epoch',ylabel='Validation accuracy (%)',title='Validation checkpoint selection')
for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
''')
    md('Training objectives differ across smoothed-label, Mixup and ordinary cross-entropy runs; their loss magnitudes are not controlled comparisons. More epochs and BF16 distinguish the Mixer recipe from the earlier residual recipe. Ordinary, EMA and SWA entries share one fine-tuning run; do not sum their wall times as independent experiments.')
    code('''from select_final import write_pareto
pareto=pd.DataFrame(write_pareto())
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,cost in zip(axes,['parameters','macs']):
    ax.scatter(pareto[cost],100*pareto.val_accuracy,c=pareto.pareto.astype(int),cmap='coolwarm',s=38)
    ax.set_title('Red: Pareto frontier (names in table below)')
    ax.set_xscale('log'); ax.set_xlabel(cost); ax.set_ylabel('Validation accuracy (%)'); ax.grid(alpha=.2)
plt.tight_layout(); plt.show()
display(pareto[pareto.pareto].sort_values('macs'))
''')
    md('Pareto dominance is defined jointly over validation accuracy (higher), physical parameters (lower) and dense MACs (lower). A model with fewer parameters is not necessarily cheaper: Mixer weights are reused over tokens. FLOPs are approximately twice dense MACs, excluding nonlinearities, normalization, memory traffic and augmentation. This chart describes single-model, single-view checkpoints, not ensemble cost.')
    code('''candidates=pd.read_csv('results/validation_candidates.csv')
raw_views=candidates[candidates['name'].str.match(r'^[A-Za-z0-9_]+:[0-9]+$')].copy()
raw_views[['model','views']]=raw_views['name'].str.rsplit(':',n=1,expand=True)
raw_views['views']=raw_views['views'].astype(int)
selected_names={c['name'] for c in json.loads(Path('results/final_recipe.json').read_text())['components']}
fig,ax=plt.subplots(figsize=(10,4))
for name,group in raw_views[raw_views['model'].isin(selected_names)].groupby('model'):
    group=group.sort_values('views')
    ax.plot(group['views'].astype(str),100*group['accuracy'],marker='o',label=name)
ax.set(xlabel='Inference views per model',ylabel='Validation accuracy (%)',title='View ablation for the selected ensemble members')
ax.grid(alpha=.2); ax.legend(fontsize=8); plt.tight_layout(); plt.show()
''')
    md('''## 4. Frozen final evaluation

    The accuracy-first search evaluates 1/2/4/10/18/30/50-view inference for completed models within 3.5 percentage points of the best raw validation score. The ten best distinct checkpoints contribute raw equal-weight pairs/triples; calibrated prefixes and greedy combinations can use every eligible checkpoint. Up to three starting models undergo eight greedy convex-weight additions, with fixed candidate weights 0.1/0.2/0.35/0.5. Each accepted greedy step must avoid reducing correct count on either fixed 3,000-image validation half. Both halves remain development data; this guard is not an independent test or a significance claim. All attempted weight settings are counted in `weighted_search.json`, including rejected ones.

    Ten views consist of the image and four one-pixel cardinal translations, each with and without horizontal flip. Eighteen views use every 3x3 translation offset with both flip states. Thirty views combine the ten-view setting with affine-grid scales 1/0.96/1.04, using bilinear interpolation; fifty views cover every 5x5 translation offset with both flip states. Padding is zero, never wraparound. The earlier 1/2/4-view definitions remain unchanged. Selection maximizes total validation correct count, with validation negative log-likelihood as its only tie-breaker. Parameters and MACs are recorded but have no selection penalty or cutoff. The fixed candidate search is exploratory and cannot guarantee a global optimum or the highest class ranking.

    Before greedy/prefix ensemble construction, each selected model's view-averaged probabilities are calibrated as `softmax(log(p)/T)`, with T chosen from 0.75/1/1.25/1.5/2 by validation NLL. This post-view transformation preserves each individual model's argmax but can change ensemble decisions by adjusting relative confidence. Raw individual, equal-pair and equal-triple candidates remain eligible. Temperature and mixture-weight choices use no test labels.

    The follow-up keeps this ensemble as its reference. Four garment/focal-loss recipes and four spatial-readout recipes add 24 ordinary/EMA/SWA checkpoints. Each round searches 583 recorded validation recipes, including reduced views and dropped members. Candidates cannot reduce Shirt F1 or correct counts on either fixed validation half; accuracy ranks first, followed by lower MACs, fewer parameters and NLL. Neither round displaced the reference.

    A final 43-candidate gate search considers each existing ensemble member, one or two first-stage views, and thresholds 0.90/0.95/0.975/0.99/0.995/0.999/0.9999. Only predicted classes 1/5/7/8/9 may exit early; all upper-garment predictions receive full inference. Validation accuracy ranks first, then average MACs and NLL, with the same Shirt F1/half-count guards. The actual routed pipeline was recomputed before freezing, including batch repacking. The selected gate gives 5,721/6,000 correct, compared with the reference's 5,717. Validation remains development data; the small gain is not a significance claim.
    ''')
    code('''recipe=json.loads(Path('results/final_recipe.json').read_text())
display(pd.DataFrame(recipe['components']))
print({k:recipe[k] for k in ['frozen_at','accuracy','correct','examples','parameters','macs','candidates']})
display(pd.read_csv('results/validation_candidates.csv').head(12))
print('Weighted search attempts:',json.loads(Path('results/weighted_search.json').read_text()))
from evaluate_final import evaluate_final
test=evaluate_final()
if not RETRAIN_ALL:
    print('Recorded accuracy:', ''' + repr(metrics['accuracy']) + ''', '; current evaluation:', test['accuracy'])
    print('Small differences can occur across CPU/FP32 and CUDA/BF16 environments.')
display(pd.DataFrame(test['classification_report']).T.round(4))
''')
    code('''from sklearn.metrics import ConfusionMatrixDisplay
test_data=FashionMNIST('data',train=False,download=True)
fig,ax=plt.subplots(figsize=(9,8))
ConfusionMatrixDisplay(np.asarray(test['confusion_matrix']),display_labels=test_data.classes).plot(ax=ax,cmap='Blues',colorbar=False,xticks_rotation=45)
ax.set_title('Frozen classifier: test confusion matrix'); plt.tight_layout(); plt.show()
predictions=np.load('results/final_test_predictions.npz')
labels=predictions['labels']; predicted=predictions['probabilities'].argmax(1)
errors=np.flatnonzero(predicted!=labels)[:12]
fig,axes=plt.subplots(2,6,figsize=(14,6.5),layout='constrained')
for ax,idx in zip(axes.flat,errors):
    ax.imshow(test_data.data[idx],cmap='gray'); ax.axis('off')
    ax.set_title(f'True: {test_data.classes[labels[idx]]}\\nPred: {test_data.classes[predicted[idx]]}',fontsize=8)
fig.suptitle('First 12 errors in test-index order (not cherry-picked)'); plt.show()
''')
    md(f"The lowest class F1 is **{weakest}: {class_metrics[weakest]['f1-score']:.4f}**. The largest directed confusions are "
       +'; '.join(f'**{true} -> {pred}: {count} images**' for count,true,pred in confusions)
       +'. These results describe the frozen classifier; they are not used to introduce another model change. Similar garment silhouettes remain a challenge at 28x28 resolution.')
    def acc(name):
        return f"{evidence[name]['val_accuracy']:.2%}"
    md(f'''## 5. Technical summary and conclusions

    ### What improved

    The fixed sigmoid/SGD baseline reaches {acc('baseline_sgd_sigmoid')} validation accuracy. Changing only its optimizer to AdamW reaches {acc('adamw_sigmoid')}, indicating that optimization and the small SGD learning rate strongly limit the reference run. The wider residual GELU recipe reaches {acc('residual_gelu')}; it changes architecture, normalization, regularization and scheduling together, so this is not an isolated causal estimate of skip connections.

    Matched residual comparisons yield SiLU {acc('residual_silu')}, BatchNorm {acc('residual_bn')}, geometry {acc('residual_geometry')}, and Mixup {acc('residual_mixup')}. Mixup does not help this GELU configuration at the tested alpha and budget. Approximately parameter-matched SwiGLU reaches {acc('residual_swiglu')}; Muon/AdamW reaches {acc('residual_muon')}. SAM {acc('residual_sam')} and ASAM {acc('residual_asam')} do not improve that baseline, despite twice the gradient evaluations. These are negative results for the tested settings, not universal method rankings.

    The matched Mixer comparison gives AdamW {acc('mixer_adamw')} and Muon/AdamW {acc('mixer_muon')}. Their observed training/validation times are {evidence['mixer_adamw']['seconds']:.1f}s and {evidence['mixer_muon']['seconds']:.1f}s on the shared GPU. Muon offers a small accuracy gain here, not a twofold speed-up. Its language-model scaling results cannot be transferred to this workload. Parameter grouping is important: only hidden matrices use Muon.

    Clean fine-tuning gives ordinary weights {acc('mixer_average')}, EMA {acc('mixer_average_ema')} and SWA {acc('mixer_average_swa')}. Late clean-training accuracy approaches 100% while validation loss grows; checkpoint selection and EMA limit this overfitting, whereas SWA is not automatically better. The matched small student obtains {acc('student_control')} without distillation and {acc('student_distill')} with distillation, using 90,525 parameters and 4,867,712 dense MACs per image. Distillation adds the teacher's pretraining and forward-pass costs; student-only deployment avoids them.

    The student control and distilled student tie on best validation accuracy. This trial therefore does not demonstrate an accuracy benefit from distillation at the tested setting, despite its additional training cost.

    ### Accuracy-first extension

    The objective is maximum validation accuracy without a computation budget. The finer-patch trial obtains ordinary/EMA/SWA validation accuracies {acc('accuracy_p2')}/{acc('accuracy_p2_ema')}/{acc('accuracy_p2_swa')}; the wider/deeper trial obtains {acc('accuracy_wide')}/{acc('accuracy_wide_ema')}/{acc('accuracy_wide_swa')}. Augmented refinement of the Muon-trained model obtains {acc('accuracy_muon_refine')}/{acc('accuracy_muon_refine_ema')}/{acc('accuracy_muon_refine_swa')}, and the large flat-input residual model obtains {acc('accuracy_residual')}/{acc('accuracy_residual_ema')}/{acc('accuracy_residual_swa')}. A large model or a finer patch is not assumed to improve accuracy; measured validation results decide.

    The finer-patch clean fine-tuning trial obtains ordinary/EMA/SWA accuracies {acc('accuracy_p2_clean')}/{acc('accuracy_p2_clean_ema')}/{acc('accuracy_p2_clean_swa')}; the corresponding wider-model trial obtains {acc('accuracy_wide_clean')}/{acc('accuracy_wide_clean_ema')}/{acc('accuracy_wide_clean_swa')}. Parent checkpoints remain eligible. These follow-up trials test whether the clean fine-tuning benefit observed in the initial four-pixel Mixer transfers to the new models.

    Enlarging the inference and ensemble search lets models combine complementary errors. The selected component table records each model, number of views, exact weight when nonuniform, and checkpoint hash. The initial accuracy-first search used validation correct count and NLL. Later deployment comparisons keep accuracy first and use lower average MACs and parameter counts to break ties, with development-half and Shirt-F1 guards; training and inference costs remain visible for assessment.

    ### Regularization and model efficiency

    The 2x2 control yields neither dropout nor decay {acc('regularization_none')}, dropout only {acc('regularization_dropout')}, decay only {acc('regularization_decay')}, and both {acc('residual_gelu')}. These results describe one fixed architecture and seed; they do not establish a universally optimal regularizer.

    Retaining 75% of channel neurons gives {acc('mixer_pruned_75_initial')} before recovery and {acc('mixer_pruned_75')} after fine-tuning; retaining 50% gives {acc('mixer_pruned_50_initial')} and {acc('mixer_pruned_50')}. The compression removes real matrix dimensions and its fine-tuning cost is additional. The Pareto table reports measured accuracy against both parameter and computation cost. Initial pruning damage and recovered accuracy are reported separately.

    ### Final result and cost

    The frozen recipe is **{recipe['name']}**. It obtains **{metrics['correct']}/10,000 correct ({metrics['accuracy']:.2%})** on the official test split, compared with **{metrics['baseline']['accuracy']:.2%}** for the fixed sigmoid/SGD reference. Macro precision is **{metrics['macro_precision']:.4f}**, macro recall **{metrics['macro_recall']:.4f}**, macro F1 **{metrics['macro_f1']:.4f}**, and weighted F1 **{metrics['weighted_f1']:.4f}**. The confusion matrix and per-class table identify remaining class-specific errors.

    Deployment requires **{recipe['parameters']:,} total stored parameters**. The actual test routing averages **{metrics['macs']:,.0f} dense MACs per image**, approximately **{2*metrics['macs']:,.0f} dense FLOPs**. The validation average is **{recipe['macs']:,.0f} MACs/image** and the worst case is **{recipe.get('worst_case_macs',recipe['macs']):,} MACs/image**. Conditional cost depends on the input distribution. The worst case includes both the first-stage pass and the full ensemble; it is slightly higher than the ungated ensemble. Shared first-stage weights are counted only once. No claim of lower parameter count, uniformly lower worst-case FLOPs, or a guaranteed tie-break award is made.

    The validation candidate table retains alternatives and their cost. No independent test evaluation of every candidate is used to choose the displayed winner. A higher validation score from a large search may reflect both genuine error complementarity and validation overfitting; small differences should not be interpreted as proven generalization improvements.

    ### Limits and next steps

    All experiments use one assignment-specific seed and one validation split. Repeated validation selection may overfit that split; architecture comparisons with different budgets and regularizers are recipe comparisons, not clean causal ablations. Shared-device wall time is descriptive, not a controlled hardware benchmark. The descriptive Wilson 95% interval for final test accuracy is [{metrics['accuracy_ci95_wilson'][0]:.2%}, {metrics['accuracy_ci95_wilson'][1]:.2%}]; it does not account for adaptive validation search, prior public-benchmark exposure, dataset shift, or prove that small differences are significant. This is an exploratory benchmark continuation, not a newly blinded evaluation. Further research should use independent seeds and a new validation protocol, rather than tuning against this test score.

    Section A is supplied separately as a typed study guide. The assignment requires the student's genuine handwritten or iPad-written answers in one PDF. A typed guide is not a compliant handwritten submission. Review and understand the answers and code, and follow the course's assistance/disclosure rules before submission.
    ''')
    md('''## 6. Targeted post-training and deployment experiments

    All post-training uses Fashion-MNIST only. The auxiliary objective adds 0.5 times conditional cross-entropy among T-shirt, Pullover, Dress, Coat and Shirt to ordinary ten-class CE; other classes remain in training to avoid forgetting. Focal loss uses gamma 1 without a class-balancing alpha. Each wide/fine-patch trial has an existing clean-refinement control. The spatial/pruned pair isolates the auxiliary-loss change within that spatial recipe. No MNIST pretraining or other external dataset was introduced.

    Single-model gains did not translate into a better ensemble. Spatial readouts overfit, and neither refinement round passed the reference's global/half-count/Shirt-F1 guards with a better result. These are negative results for these settings, not a claim that targeted post-training is universally ineffective. The gate improves validation Shirt precision slightly; it does not demonstrate a substantial improvement in Shirt recall.

    The following deployment numbers are **recorded measurements**, not rerun timings in this notebook. NVFP4 was tested on the ungated reference ensemble, using torchao 0.16.0 with PyTorch 2.10.0 on RTX 5090. Only channel hidden matrices were quantized to packed W4A4; token matrices, embeddings, normalization and heads retained higher precision. Profiler evidence includes the native scaled GEMM and SM120 E2M1 CUTLASS kernel. NVFP4 was slower and less accurate here, so the submitted model retains original precision. Optional benchmark scripts are included in the bundle; torchao is not required for default Run All.
    ''')
    code('''nv=json.loads(Path('results/nvfp4_benchmark.json').read_text())
deployment=pd.DataFrame([{'mode':mode,'validation_accuracy_%':100*nv['validation'][mode]['accuracy'],
                          'Shirt_F1':nv['validation'][mode]['shirt_f1'],
                          'milliseconds_per_128_images':timing['full_recipe_batch128']['median_ms']}
                         for mode,timing in nv['timing'].items()])
display(deployment.round(4))
display(pd.DataFrame(nv['members'])[['name','reference_storage_bytes','all_bf16_parameter_bytes','nvfp4_storage_bytes']])
print('Real FP4 operations:',nv['fp4_profiler_operations'])
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
labels=['Original','NVFP4','Original compiled','NVFP4 compiled']
axes[0].bar(labels,deployment['validation_accuracy_%']); axes[0].set(ylabel='Validation accuracy (%)',ylim=(94,96))
axes[1].bar(labels,deployment['milliseconds_per_128_images']); axes[1].set(ylabel='Milliseconds / 128 images (lower is better)')
for ax in axes: ax.tick_params(axis='x',rotation=20)
fig.suptitle('Recorded ungated reference benchmark on shared RTX 5090'); plt.show()
speed=json.loads(Path('results/cascade_benchmark.json').read_text())
display(pd.DataFrame([{'pipeline':name,'milliseconds_per_1024_images':row['median_ms'],
                       'images_per_second':row['images_per_second'],'average_MACs':row['execution']['macs_per_image']}
                      for name,row in speed['timing'].items()]))
print('Recorded cascade speedup:',speed['speedup'])
print('Actual current test routing:',test['execution'])
''')
    md('''Quantization reduces some byte storage, not logical parameter count or mathematical dense FLOPs. The all-BF16 byte column is a storage comparison, not the original FP32 checkpoint size; scale/layout overhead can make partial NVFP4 storage exceed an all-BF16 model for small shapes. Compilation changes floating-point arithmetic slightly: the original compiled path loses one validation-correct example, so it is not described as lossless. The frozen submitted path uses eager original-precision inference. Cascade timing uses 1,024 validation images and the same preloaded weights in both paths; NVFP4 timing uses batches of 128. Do not directly compare these different batch totals. All timings exclude load/compile time and are descriptive measurements on a shared device.''')
    md('''## 7. Input processing and activation-aware weight quantization

    These are recorded validation-only follow-ups with the selected cascade held fixed. Five mild input changes were compared with the original input: gamma 0.95/1.05, intensity gain 0.95/1.05, and center-of-intensity alignment bounded to half a pixel. Processing precedes the existing geometric TTA. None improved accuracy or Shirt F1, so none was adopted. These are input preprocessing operations; temperature scaling and the confidence gate operate on classifier outputs. One shared temperature preserves a single classifier's argmax, although component-specific temperatures can affect ensemble decisions and confidence thresholds.

    Native torchao AWQ uses activation statistics from 100 training images (ten per class), 20 scale candidates and group-32 INT4 channel weights with BF16 activations. Both plain INT4 and a channel-BF16 control use the same layer subset. The native observers capture activations before autocast; saved observations are cast to the actual BF16 GEMM input dtype before the offline search. Token mixing, embeddings, normalization and heads retain original precision. No validation images calibrate weights, and no new test evaluation selects a candidate.

    The table below reports **recorded eager measurements**, not a new timing run. All paths use the same 1,024 validation images, three warmups and seven synchronized timing repetitions. Loading and calibration are excluded. The real packed INT4 kernel is verified by the profiler. This particular backend pads input widths to multiples of 1,024, creating substantial overhead for the small channel matrices. It is not representative of every AWQ backend, larger model or compiled implementation. Logical parameters and mathematical dense FLOPs do not decrease.
    ''')
    code('''processing=json.loads(Path('results/image_processing.json').read_text())
display(pd.DataFrame(processing['candidates'])[['name','correct','accuracy','shirt_recall','shirt_f1']])
awq=json.loads(Path('results/awq_benchmark.json').read_text())
assert processing['complete'] and awq['complete']
assert not processing['eligible_improvements']
awq_rows=[{'mode':mode,'validation_accuracy_%':100*row['accuracy'],
           'Shirt_F1':row['shirt_f1'],'milliseconds_per_1024_images':awq['timing'][mode]['median_ms'],
           'stored_tensor_bytes':sum(m['storage_bytes'][mode] for m in awq['members'])}
          for mode,row in awq['validation'].items()]
display(pd.DataFrame(awq_rows).round(4))
print('Packed INT4 operations:',awq['packed_profiler_operations'])
print('Input-processing and AWQ candidates were not adopted.')
''')
    md('''AWQ obtains 5,711/6,000 correct versus 5,721 for the original cascade, and is approximately 3.52 times slower in this eager backend. Its 7,331,088 stored tensor bytes improve on the original FP32 storage but exceed the 5,422,224-byte channel-BF16 control after padding and scales. Plain INT4 also outperforms AWQ in this particular calibration trial; activation-aware scaling is not guaranteed to improve classification accuracy. The channel-BF16 control matches original validation metrics and is faster here, but remains a benchmark control rather than a separately promoted submission recipe. That comparison retained the 94.42% reference model; the following architecture experiments use it as their frozen comparison.''')
    md('''## 8. Detector-inspired pure MLP experiments

    These trials borrow feature-fusion, partial-channel processing and intermediate-supervision ideas associated with modern detectors. They are **not YOLO models or reproductions of PGI**. All spatial operations are reshapes, concatenations and dense Linear layers; there is no convolution, attention, external pretraining, detection-box loss or NMS.

    A 2x2 Linear patch embedding creates 196 tokens of width 96. Three fine-stage Mixer blocks process them. Each adjacent 2x2 group of tokens is concatenated, normalized and linearly projected to width 192, giving 49 coarse tokens. Three coarse Mixer blocks follow. The control classifies the pooled coarse features; feature fusion concatenates pooled fine and coarse features before its small Linear head. Backbone initialization and the post-initialization random-number stream are identical for that pair.

    Auxiliary supervision adds a ten-class head on pooled fine features with loss weight 0.3. It is optimized during training only and is absent from deployment checkpoints and inference FLOPs. Its construction preserves the paired model's random-number stream. This tests ordinary deep supervision, not the full programmable-gradient method from YOLOv9.

    The partial-channel variant passes one half of the normalized channels directly to concatenation and transforms the other half through an MLP, followed by a Linear fusion layer. Residual connections and full token mixing remain. This changes architecture and parameter count, so matching the seed and training hyperparameters does not establish a parameter-matched causal effect.

    Each initial trial uses 120 epochs with the same AdamW/warmup/cosine, geometric augmentation, label smoothing, dropout and weight decay. Every architecture then receives 30 clean-refinement epochs from its best raw-validation ordinary/EMA/SWA checkpoint. Optimizers restart for refinement; the auxiliary trial also constructs a fresh training-only head because that head is excluded from deployment checkpoints. The table reports the best raw-validation checkpoint from those six candidates per architecture. Training-only auxiliary parameters are reported separately from deployment cost.
    ''')
    code('''yolo=json.loads(Path('results/yolo_summary.json').read_text())
assert yolo['complete']
display(pd.DataFrame(yolo['families'])[['family','selected_checkpoint','accuracy','shirt_recall','shirt_f1','parameters','macs']])
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
families=pd.DataFrame(yolo['families'])
labels=['Coarse control','Feature fusion','Auxiliary loss','Partial channels']
axes[0].plot(labels,100*families['accuracy'],'o'); axes[0].set(ylabel='Single-view validation accuracy (%)',ylim=(92,95)); axes[0].grid(axis='y',alpha=.25)
axes[1].bar(labels,families['macs']/1e6); axes[1].set(ylabel='Million dense MACs per image')
for ax in axes: ax.tick_params(axis='x',rotation=20)
fig.suptitle('Recorded pure-MLP architecture comparisons; equal epoch budgets'); plt.show()
print('Cascade validation correct:',yolo['reference_correct'],'->',yolo['selected_correct'])
print('Finite validation candidate count:',yolo['candidate_count'])
display(pd.DataFrame(yolo['selected_components']))
''')
    md('''Deployment selection keeps the original confidence gate fixed. One raw-validation checkpoint per architecture enters a bounded comparison of 1/10/30 views, standalone inference, additions with weights 0.1/0.2/0.35, and replacement of either non-gate ensemble member. Selection prioritizes correct count, then average MACs, parameters and NLL; neither development-half correct count nor Shirt F1 may regress. The chosen recipe is recomputed with actual routing before freezing. The halves are reused development data, not independent holdouts. Small gains after many validation comparisons are exploratory and do not establish statistical significance. Only individually supported changes should be combined; negative comparisons remain visible.''')
    md('''The original 5,721/6,000-correct cascade wins all 77 comparisons; the best non-reference candidate obtains 5,714. Therefore the final submission remains the previously frozen 94.42% test model. Feature fusion and auxiliary supervision do not improve the coarse-only control in this experiment. Partial-channel fusion is smaller and slightly more accurate than full-channel fusion, but its Shirt F1 is lower and it does not improve the existing deployment. Auxiliary supervision did not justify stacking it with the partial-channel variant, so that additional combination was not run. All 91 saved project checkpoints, including the 24 new ones, were reloaded and their recorded validation accuracy and deployment costs verified. These negative results apply to this configuration, budget and seed; they do not invalidate the broader design ideas.''')
    md('''## 9. PIL-inspired closed-form classification heads

    Motivated by Guo et al.'s pseudoinverse-learning review, this trial freezes each selected MLP's features and refits only its existing Linear classifier. It is a hybrid of gradient-trained features and a closed-form readout, not a reproduction of full-network PIL or a claim that all network weights have a one-step global solution.

    The objective is `mean(||H W + b - one_hot(y)||^2) + lambda * ||W||^2`, where the mean is over training images and the squared norm sums over classes. Centering features and targets gives an unpenalized intercept. CPU float64 thin SVD solves the problem without explicitly forming a pseudoinverse or normal equations. Lambda zero uses a standard dimension-scaled singular-value tolerance. Eight fixed penalties [0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1] are compared; all coefficients are fitted on 54,000 training images only. Validation correct count, then NLL, selects the head. One-hot targets use a linear output; no arctanh is applied.

    Features are rounded to the BF16 input dtype actually used by the classifier. Cached head predictions must match full-model predictions exactly, and every non-head saved tensor must remain bitwise unchanged. Deployment parameters and MACs are unchanged for a replacement head. Readout fitting has zero gradient epochs, but the original backbone-training cost is still required and is excluded from the recorded fitting time.
    ''')
    code('''pil=json.loads(Path('results/pil_experiment.json').read_text())
assert pil['complete']
pil_rows=[{'model':h['parent'],'original_accuracy_%':100*h['original']['accuracy'],
           'ridge_accuracy_%':100*h['selected']['accuracy'],'lambda':h['selected']['penalty'],
           'original_Shirt_F1':h['original']['shirt_f1'],'ridge_Shirt_F1':h['selected']['shirt_f1'],
           'parameters':h['parameters'],'MACs':h['macs']} for h in pil['heads']]
display(pd.DataFrame(pil_rows).round(5))
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
for h in pil['heads']:
    ax.plot([r['penalty'] for r in h['candidates']],
            [100*r['accuracy'] for r in h['candidates']],'o-',label=h['parent'])
ax.set(xscale='symlog',xlabel='Ridge penalty (0 = pseudoinverse)',ylabel='Single-view validation accuracy (%)')
ax.set_xscale('symlog',linthresh=1e-6); ax.legend(fontsize=8); ax.grid(alpha=.25)
ax.set_title('Recorded validation comparison; coefficients fitted on training data only'); plt.show()
print('Deployment candidates:',pil['deployment_candidates'])
print('Reference / selected validation correct:',pil['reference_correct'],pil['selected_correct'])
from pil_experiment import self_check as check_ridge
check_ridge()
''')
    md('''The wide member improves from 5,642 to 5,651 correct (94.03% to 94.18%), the fine-patch member from 5,631 to 5,634 (93.85% to 93.90%), and the pruned member stays at 5,622 (93.70%). Wide-model Shirt F1 slightly declines. These small validation changes are descriptive single-split results, not evidence of statistical significance.

    Squared-error scores have a different confidence scale from cross-entropy logits. Each refitted member therefore selects a post-view temperature from [0.05, 0.1, 0.2, 0.5, 1, 2] by validation NLL; all choose 0.1. The original gate and original members' temperatures remain fixed. The 16 deployment comparisons include the original cascade, three standalone refits with original view counts, nine additions (weights 0.1/0.2/0.35), and three combinations replacing either or both non-gate members. Additional members execute full extra backbones and are counted accordingly; no unsupported feature-sharing speedup is claimed.

    The original cascade wins with 5,721/6,000 correct; the best alternative reaches 5,710. Both development-half correct counts and Shirt F1 are guarded. The new heads are retained as experimental checkpoints, while the submitted model, original freeze timestamp and previously measured 94.42% test score stay unchanged. No new candidate test evaluation is used for selection. Repeated validation reuse remains a limitation.''')
    md('''## 10. Fixed image features for difficult garment classes

    Three paired trials retain grayscale and optionally append signed horizontal/vertical central differences, then grayscale minus its 3x3 local mean. Replicate padding avoids artificial edges on a constant background. These deterministic features are computed after any geometric transformation for each training or inference view. They introduce no fitted statistics or trainable convolution; all trainable feature mixing remains Linear MLP layers.

    All three trials start from the same wide SWA checkpoint and receive 30 clean fine-tuning epochs, AdamW at 5e-5, decay 0.01, identical cosine warmup schedule, seed and dropout/shuffle random stream. Additional input-projection columns start at zero, preserving the grayscale function in real arithmetic. A runnable check verifies feature direction, constant boundaries, initial FP32 logits and finite gradients. BF16 kernels can have shape-dependent rounding, so training is not claimed bitwise identical.

    Ordinary, EMA and SWA checkpoints compete within each family on validation accuracy then NLL. Deployment uses the same bounded 1/10/30-view comparison and Shirt-F1/development-half guards as the detector-inspired trials. Test labels are not used. Dense MACs include the wider input projection; fixed preprocessing adds about 3,136 scalar operations per image for gradients or 10,976 for gradients plus local contrast, excluding memory operations. These counts are not latency measurements.
    ''')
    code('''features=json.loads(Path('results/features_summary.json').read_text())
feature_details=json.loads(Path('results/features_details.json').read_text())
assert features['complete'] and feature_details['complete']
display(pd.DataFrame(features['families'])[['family','selected_checkpoint','accuracy','shirt_recall','shirt_f1','parameters','macs']])
print('Reference / selected cascade validation correct:',features['reference_correct'],features['selected_correct'])
print('Bounded deployment candidates:',features['candidate_count'])
from feature_experiment import self_check as check_features
check_features()
garments=[0,2,4,6]
garment_labels=['T-shirt','Pullover','Coat','Shirt']
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,row in zip(axes,feature_details['rows']):
    counts=np.asarray(row['confusion_matrix'])[np.ix_(garments,garments)]
    ax.imshow(counts,cmap='Blues',vmin=0,vmax=600)
    for i in range(4):
        for j in range(4): ax.text(j,i,str(counts[i,j]),ha='center',va='center',color='white' if counts[i,j]>300 else 'black')
    ax.set(xticks=range(4),yticks=range(4),xticklabels=garment_labels,yticklabels=garment_labels,
           xlabel='Predicted class',ylabel='True class',title=row['name'].replace('features_',''))
fig.suptitle('Recorded validation confusion: 600 images per true class; other classes omitted'); plt.show()
''')
    code('''from models import image_features
from train import load_data
from evaluation import load_model, probabilities
train_x,train_y,validation_x,validation_y,_,_=load_data()
del train_x,train_y
parent_probs=probabilities(load_model('accuracy_wide_clean_swa'),validation_x,1)
ids=torch.where((validation_y.cpu()==6)&(parent_probs.argmax(1)!=6))[0][:3]
maps=image_features(validation_x[ids.to(validation_x.device)],'contrast').cpu()
fig,axes=plt.subplots(len(ids),4,figsize=(9,7),squeeze=False,layout='constrained')
for i in range(len(ids)):
    for j,title in enumerate(['Original grayscale','Horizontal gradient','Vertical gradient','Local contrast']):
        axes[i,j].imshow(maps[i,j],cmap='gray' if j==0 else 'coolwarm',vmin=0 if j==0 else -.5,vmax=1 if j==0 else .5)
        axes[i,j].set_title(title if i==0 else '',fontsize=10); axes[i,j].axis('off')
fig.suptitle('First three validation Shirts misclassified by the parent; illustrative, not proof of improvement')
plt.show()
del validation_x,validation_y
''')
    md('''The grayscale and gradient variants both obtain 5,642/6,000 correct (94.0333%); adding local contrast gives 5,640 (94.00%). Correctly classified Shirts fall from 485/600 to 484 and 483. Gradient Shirt F1 rises slightly because false-positive predictions decrease, not because recall improves. Across 58 deployment candidates, the strongest non-reference alternative reaches 5,718 correct and the original 5,721-correct cascade remains selected. The submitted weights and original freeze timestamp are unchanged; no new candidate test evaluation is performed.

    Explicit gradients highlight outlines and local contrast emphasizes small intensity changes, but neither adds information absent from the pixels. These linear filters can be represented by a sufficiently capable network; their practical effect is to change the available input representation and optimization. They may also amplify noise or remove useful texture if substituted for grayscale, which is why grayscale is retained. This single-seed, short fine-tuning comparison does not settle performance from scratch or under longer training; repeated validation reuse limits generalization claims.''')
    md('''## 11. Shared-trunk sparse MLP experts

    This experiment tests whether routing among three small residual MLP experts improves the wide Mixer's pooled representation. Each expert maps 192 -> 384 -> 192 with GELU; its output is added to the shared feature vector before the original classifier. A Linear router maps 192 features to three probabilities. The approximately total-parameter-matched control has one 192 -> 1152 -> 192 expert and a redundant one-output router for implementation consistency.

    Both trials start from the same wide SWA checkpoint and fine-tune all weights for 30 clean epochs with the same AdamW schedule. Expert output matrices and biases start at zero. Additional initialization preserves the trunk random stream; experts have no dropout so the shared-trunk dropout/shuffle stream remains matched. Training uses soft weighted expert features, while validation and deployed inference use argmax top-1 routing and execute only the selected expert on each image. This train/inference difference can hurt accuracy, so dense soft-inference metrics are also recorded for the same checkpoints.

    No load-balancing loss is used; route counts expose unequal utilization. These shared-trunk experts are not independently bootstrapped bagging models, class-supervised garment experts, or speculative decoding. The expensive shared trunk still runs for every image. Parameter counts include all stored experts, while sparse MACs include the trunk, router, selected expert and head. Equal-sized experts make sparse dense-Linear MACs independent of the route. The fixed feature preprocessing experiments remain separate.
    ''')
    code('''moe=json.loads(Path('results/moe_experiment.json').read_text())
moe_selection=json.loads(Path('results/moe_summary.json').read_text())
assert moe['complete'] and moe_selection['complete']
display(pd.DataFrame([{k:r[k] for k in ['name','accuracy','shirt_recall','shirt_f1','parameters','sparse_macs','full_expert_macs','routes']} for r in moe['checkpoints']]))
display(pd.DataFrame([{'name':r['name'],'top1_accuracy':r['accuracy'],'soft_accuracy':r['dense_metrics']['accuracy']} for r in moe['checkpoints'] if 'dense_metrics' in r]))
print('Reference / selected deployment correct:',moe_selection['reference_correct'],moe_selection['selected_correct'])
print('Deployment candidate count:',moe_selection['candidate_count'])
from moe_experiment import self_check as check_moe
check_moe()
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
axes[0].bar(['Expert 1','Expert 2','Expert 3'],moe['selected']['routes'])
axes[0].set(title='Selected top-1 checkpoint: validation routes',ylabel='Images')
timings=moe['timings']
axes[1].barh(list(timings),[r['median_ms'] for r in timings.values()])
axes[1].set(xlabel='Median milliseconds / 1,024 images',title='Recorded eager latency; shared GPU')
axes[1].tick_params(axis='y',labelsize=8)
plt.show()
''')
    md('''Latency uses the same first 1,024 validation images in batches of 128, three warmups and seven synchronized timing repetitions. Timing includes routing and model computation but excludes loading, preprocessing transfer, softmax and output transfer; these are local forward-only comparisons, not end-to-end cascade timings. Hardware is shared, so measurements are descriptive. Dense forward MACs do not represent backward/optimizer training FLOPs and exclude activation, normalization, memory traffic and dispatch overhead.

    One ordinary/EMA/SWA checkpoint per family enters the fixed standalone/addition/replacement comparison at 1/10/30 views. The original gate is retained and shared with its existing member. Actual routed inference verifies the winning recipe, with the existing Shirt-F1 and both-development-half guards. These are repeatedly reused development data; neither a few extra correct predictions nor route imbalance establishes a general superiority or failure of MoE.''')
    md('''The selected top-1 MoE reaches 94.0167% single-view validation accuracy, versus 94.0333% for the one-expert control. Its dense soft path has the same 5,641-correct count. All 39 deployment comparisons retain the original cascade. Median forward-only latency per 1,024 images is 7.81 ms for the parent, 9.30 ms for the control, 10.93 ms for sparse MoE and 8.65 ms for dense soft MoE. Top-1 saves only about 0.38% of full-model dense MACs relative to all-expert inference; dispatch overhead outweighs that saving in this implementation. The experiment does not support promoting these lightweight experts for accuracy or speed.''')
    md('''## 12. Stronger focal loss and mild garment weighting

    The earlier focal-gamma-one and conditional-garment losses did not improve the final recipe. A separate paired follow-up tests gamma two and class-weighted cross-entropy against the completed grayscale CE control. All start from the same wide SWA checkpoint and use the same 30-epoch clean fine-tuning recipe. Only the loss changes; each family selects ordinary/EMA/SWA by validation correct count then NLL. These new results should not be interpreted as an isolated gamma-one-versus-gamma-two comparison, because the historical gamma-one trial had a different parent.

    Focal loss is `mean((1-p_true)^2 * CE)` with no alpha balancing. Weighted CE gives weight 1.5 to T-shirt, Pullover, Coat and Shirt, and 1 to the other six classes; it divides summed weighted losses by summed sample weights, matching native PyTorch. The dataset is balanced: these weights emphasize difficult classes rather than correct a count imbalance. Focal loss also changes effective gradient magnitude at the same learning rate; no separate learning-rate sweep is performed. Neither objective guarantees increased recall or unchanged performance on other classes.

    Before seeing results, local-augmentation stacking was made conditional on a loss improving overall validation correct count without reducing Shirt recall, Shirt F1, or either development-half count versus matched CE. Aggressive upper-half crops can destroy useful garment-length evidence; training the shared network on only four classes can cause forgetting. Such changes are not assumed harmless or automatically stacked. The bounded deployment grid still includes the unchanged original cascade.
    ''')
    code('''loss_trials=json.loads(Path('results/loss_experiment.json').read_text())
loss_selection=json.loads(Path('results/loss_summary.json').read_text())
assert loss_trials['complete'] and loss_selection['complete']
display(pd.DataFrame([{k:r[k] for k in ['name','correct','accuracy','shirt_precision','shirt_recall','shirt_f1','validation_half_correct']} for r in loss_trials['selected_families']]))
print('Eligible for local-augmentation stacking:',loss_trials['supported_for_local_augmentation'])
print('Original / selected deployment correct:',loss_selection['reference_correct'],loss_selection['selected_correct'])
classes=[0,2,4,6]
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
for row in loss_trials['selected_families']:
    ax.plot(range(4),[100*row['classes'][i]['recall'] for i in classes],'o-',label=row['name'])
ax.set(xticks=range(4),xticklabels=['T-shirt','Pullover','Coat','Shirt'],ylabel='Validation recall (%)',title='Recorded paired loss comparison; 600 images per class')
ax.legend(fontsize=8); ax.grid(alpha=.25); plt.show()
''')
    md('''Matched CE gives 5,642 correct, focal gamma 2 gives 5,644, and weighted CE gives 5,641. Focal leaves Shirt recall unchanged at 485/600, while precision improves and F1 changes from 0.824830 to 0.827645. Its development-half correct counts move from [2,808, 2,834] to [2,806, 2,838], failing the predeclared no-regression guard. Weighted CE reduces Shirt correct count to 483. Neither loss qualifies for stacking local augmentation; no such combined run is claimed. All 58 deployment comparisons retain the original 5,721-correct cascade, preserving the original weights, freeze timestamp and 94.42% historical test result. These small validation differences do not establish statistical significance.''')
    md('''## 13. Faster inference using the earlier efficient MLPs

    The earlier two width-128, depth-six, patch-four Mixers used ten views each: 94.16% historical test accuracy, 957,280 stored parameters and 580,060,160 MACs per image. Import changes tensor names only; all 6,000 validation logits were bitwise equal to the original implementation. These are this student's same-data training runs, not external pretraining. `legacy_import.json` retains the original source hashes and configurations; `import_legacy.py` documents the conversion and requires the original local source folder only if re-importing.

    No additional training is needed. A bounded screen compares nine single/dual legacy first stages, three fallback recipes, six thresholds, two allowed-class sets and fixed blends, yielding 354 candidates. Twelve finalists are recomputed with actual subset routing. The initial MAC-only choices were slightly slower than their references, so they were rejected as simultaneous speed improvements. Before any new test access, selection was explicitly amended to require at least 5% lower measured median latency, lower average MACs, and no reduction in validation correct count, Shirt F1 or either reused development-half count. This amendment and the initial failed choices remain in the evidence. It is reused validation, not an independent confirmation or an untouched preregistration.

    **Balanced:** run both legacy models once; exit only when their predictions agree and mean confidence is at least 0.90. Otherwise use the original cascade. **Accuracy priority:** run the legacy Muon model on the original and flipped image; exit when averaged confidence is at least 0.95, otherwise use the original cascade. All predicted classes may exit; no true labels enter routing. Both endpoints were hash-frozen and committed before their test evaluations. Their names reflect validation objectives, not a ranking chosen from test outcomes.

    Every fallback computation actually runs only on routed images. Repeated views are recomputed and charged. Total parameters count every stored model once, including inactive fallback members. MACs count dense Linear arithmetic; they exclude normalization, activations, routing and memory overhead. Worst-case cost is 3,319,207,680 MACs for either option. Both add stored parameters, so neither dominates all references on every complexity measure.
    ''')
    code('''from hybrid_experiment import evaluate_frozen, self_check as check_hybrid
check_hybrid()
hybrid=json.loads(Path('results/hybrid_experiment.json').read_text())
recorded_hybrid=json.loads(Path('results/hybrid_test_metrics.json').read_text())
if not RETRAIN_ALL:
    hybrid_test=evaluate_frozen()
    for goal,row in hybrid_test['selected'].items():
        print(goal, 'recorded / executed test correct:', recorded_hybrid['selected'][goal]['correct'], row['correct'])
else:
    hybrid_test=recorded_hybrid
    print('Hybrid table is recorded evidence only: retrained fallback weights require a new validation freeze.')
legacy=json.loads(Path('results/legacy_import.json').read_text())
comparison=[]
for label,key,test_accuracy,test_macs in [
    ('Earlier dual 10-view','legacy',legacy['historical_test_metrics']['accuracy'],580060160),
    ('Original cascade','current',test['accuracy'],test['macs'])]:
    ref=hybrid['references'][key]
    comparison.append(dict(model=label,validation_accuracy=ref['accuracy'],test_accuracy=test_accuracy,
                           average_test_MACs=test_macs,parameters=ref['parameters'],
                           latency_ms_1024=hybrid['latency_finalists'][ref['label']]['median_ms']))
for goal,row in hybrid_test['selected'].items():
    frozen=hybrid['selected'][goal]
    comparison.append(dict(model='Hybrid '+goal,validation_accuracy=frozen['accuracy'],test_accuracy=row['accuracy'],
                           average_test_MACs=row['execution']['macs_per_image'],parameters=row['parameters'],
                           latency_ms_1024=frozen['measured_latency']['median_ms']))
comparison=pd.DataFrame(comparison)
display(comparison)
for goal,row in hybrid_test['selected'].items():
    print(goal, 'actual test routing:',row['execution'])
    display(pd.DataFrame(row['classification_report']).T.round(4))
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for ax,cost,scale,xlabel in zip(axes,['average_test_MACs','latency_ms_1024'],[1e6,1],
                               ['Average test MACs (millions)','Recorded median ms / 1,024 validation images']):
    for i,row in comparison.iterrows():
        ax.scatter(row[cost]/scale,100*row['test_accuracy'],label=row['model'],s=65)
    ax.set(xlabel=xlabel,ylabel='Observed test accuracy (%)')
    ax.grid(alpha=.25); ax.legend(fontsize=8)
fig.suptitle('Two frozen alternatives: average compute and latency trade-offs')
plt.show()
''')
    md('''Both endpoints obtain 9,449/10,000 test correct (94.49%), versus 9,416 for the earlier dual model and 9,442 for the original cascade. Balanced average test MACs are 492,477,669, down 15.1% versus the earlier dual model; accuracy-priority average test MACs are 841,892,537, down 50.4% versus the original cascade. Balanced stores 3,164,836 parameters and accuracy priority stores 2,686,196, compared with 957,280 and 2,207,556 for their respective references. Balanced is faster and cheaper on average; accuracy priority uses less storage and preserves the higher validation score. Neither option was retuned after seeing these test results.

    The same-device latency comparison uses the first 1,024 validation images, internal batch size 128, three warmups and seven synchronized repetitions. It includes views, routing, softmax, averaging and CPU output, excluding model loading and input transfer. Recorded medians are 104.39 ms for the earlier dual model, 236.45 ms for the original cascade, 95.49 ms for balanced and 144.84 ms for accuracy priority. These are shared RTX 5090 measurements selected on validation timing, not independent speed confirmation or single-image latency. Average test MACs and validation timing use different image sets and are explicitly labelled.

    The 0.07-percentage-point test gain over the original cascade is seven images, not evidence of statistical significance. The public test set and development split were repeatedly observed during the wider project; neither a class ranking nor a general improvement is guaranteed. All source code, both frozen recipes and their five unique model checkpoints are embedded for default execution. The original `final_recipe.json` remains an explicit reproducible reference; optimized deployment uses `hybrid_experiment.predict` with either `hybrid_*_recipe.json`.''')
    md('''## 14. Progressive exits and batched fallback views

    Further optimization first tested 360 fallback-view/threshold combinations with all weights fixed. None preserved the relevant reference's validation correct count, Shirt F1 and both development-half counts while reducing average MACs by at least 5%. Lowering fallback quality hurts precisely the difficult images still needing it; no candidate from that phase was tested or adopted.

    A second phase tried 24 one-pass pre-exit policies. Three actual balanced finalists preserved validation metrics and lowered compute, but their roughly 2% speed gains missed the predeclared 5% latency target. The first prediction is reused for unresolved images: only the missing second model or flipped view runs. All skipped/reused work is accounted for, with executable row-count checks. No accuracy-priority candidate qualified. These negative results remain in `refine_experiment.json` and `progressive_experiment.json`.

    A third phase batches two or four fallback views per model call for those three accuracy/compute-qualified actual finalists, yielding six additional runtime candidates. Batching improves device utilization and reduces kernel launches without changing mathematical MACs; progressive skipping provides the MAC reduction. Since BF16 rounding can change with matrix batch shape, every candidate's actual routed validation outputs are recomputed. Selection preserves correct count, Shirt F1 and both halves, requires at least 1% fewer average MACs with no added parameters, and requires at least 5% lower median latency on BOTH first and last 1,024 validation images. The one-percent compute floor was declared before the progressive experiment because it targets one inexpensive first-stage pass. Selection uses a bounded top-five shortlist, not proof of global optimality.

    **Frozen pipeline:** run `legacy_clean` once. Accept any predicted class at confidence >=0.99. Otherwise run `legacy_muon` once and average it with the saved first prediction; accept only agreement and mean confidence >=0.90. Remaining images enter the unchanged original cascade, where up to four views are concatenated per model forward. The same five models are stored, and no architecture or optimizer changes are made. Four-view groups use at most 512 transformed images for an internal batch of 128 original images, so temporary activation memory grows. This is device-specific throughput optimization, not a single-image or CPU speed guarantee.

    The selected recipe was committed before test evaluation. Test accuracy remains 9,449/10,000 (94.49%), average test MACs decrease from 492,477,669 to 475,709,806 (3.4%), and stored parameters stay at 3,164,836. Worst-case cost remains 3,319,207,680 MACs. Exactly 3,977 test images exit after the first small model, and 8,552 exit before the fallback cascade. The correct count is unchanged, but the prediction vectors and which images are correct need not be identical. No threshold was retuned from test results.
    ''')
    code('''from refine_cascade import evaluate as evaluate_refinement
refined_recipe=json.loads(Path('results/batched_balanced_recipe.json').read_text())
refined_record=json.loads(Path('results/batched_test_metrics.json').read_text())
if not RETRAIN_ALL:
    refined_test=evaluate_refinement('batched')
    print('Recorded / executed test correct:',refined_record['selected']['balanced']['correct'],refined_test['selected']['balanced']['correct'])
else:
    refined_test=refined_record
    print('Recorded refinement only: new weights require a new validation-only freeze.')
row=refined_test['selected']['balanced']
display(pd.DataFrame(row['classification_report']).T.round(4))
print('Actual test execution:',row['execution'])
display(pd.DataFrame([{'slice':label,'reference_ms':a['median_ms'],'progressive_batched_ms':b['median_ms'],
                      'time_reduction_%':100*(1-b['median_ms']/a['median_ms'])}
                     for label,a,b in zip(['First 1024','Last 1024'],refined_recipe['reference_latency'],refined_recipe['timings'])]))
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].bar(['Original balanced','Progressive + batching'],
            [refined_recipe['reference_latency'][0]['median_ms'],refined_recipe['timings'][0]['median_ms']])
axes[0].set(ylabel='Median milliseconds / 1,024 validation images',title='Recorded same-session throughput comparison')
axes[1].bar(['Original balanced','Progressive + batching'],
            [hybrid_test['selected']['balanced']['execution']['macs_per_image']/1e6,row['execution']['macs_per_image']/1e6])
axes[1].set(ylabel='Average test MACs (millions)',title='Both frozen recipes: 94.49% test accuracy')
plt.show()
''')
    md('''Observed latency changes from 95.36 to 58.09 ms on the first 1,024 validation images (39.1% lower) and from 96.36 to 56.61 ms on the last 1,024 (41.2% lower). Both sets are repeatedly reused development data and timing-selection inputs, not independent confirmation. Measurement includes routing, views, probability aggregation and CPU output, excluding loading/input transfer, with three warmups and seven synchronized repetitions on a shared RTX 5090. The original reference and both prior hybrids remain available for comparison. Use `hybrid_experiment.predict` with `results/batched_balanced_recipe.json` for this deployment; `python3 refine_cascade.py --batched --test` evaluates its frozen weights.

    **Optimizer scope.** The user's “pre head Muon” likely refers to **Per-Head Muon**. Kimi K3's official report, Section 2.5, partitions attention Q/K/V momentum matrices along head boundaries and orthogonalizes each block separately. It targets training dynamics and optimizer overhead, not extra inference heads or lower deployed MACs. Our pure MLPs have no attention heads; that variant is not implemented or claimed. Existing Muon training already updates hidden-block matrices with Muon and embedding/classifier/bias/normalization parameters with AdamW.

    **Reward versus supervised feedback.** RL rewards can encode error severity and compute costs, but they must be designed; RL does not inherently diagnose mistakes or know corrective actions. Cross-entropy already provides graded confidence feedback. A learned routing policy could choose between exit and additional computation, with a reward reflecting classification loss and cost. This round uses explicit validation-selected thresholds, not RL, a contextual-bandit training run or a newly trained gate. No benefit from those untested alternatives is claimed.''')
    md('''## 15. Predicted-class thresholds and repeated-view reuse

    A single exit threshold need not allocate computation equally well across predicted classes. With all weights, the .99 first-pass pre-exit, four-view grouping and fallback fixed, compare 36 tuples: upper garments (T-shirt, Pullover, Coat, Shirt) use .85/.90/.95/.975; Dress uses .80/.90/.95; remaining classes use .70/.80/.90. These groups are selected by the model's predicted class, never the true label. The two small models must still agree before the second exit. This is a finite threshold policy, not a learned router or RL experiment.

    Preserve validation correct count, Shirt F1 and both fixed development halves; require at least 1% fewer average MACs, no extra stored parameters, and at least 5% lower median latency on both first/last 1,024 validation images. Cached screening is followed by actual routed recomputation of the top five eligible candidates. The selected thresholds are **.85 for predicted upper garments and .90 for all other classes**. Validation moves from 5,717 to 5,718 correct, Shirt F1 from 0.859348 to 0.859829, and half counts from [2,855,2,862] to [2,856,2,862]. Average validation MACs decrease from 460,521,803 to 399,102,505. One extra validation-correct image does not establish a general accuracy improvement.

    Before any new test access, a second phase tested four runtime variants: reuse the wide fallback gate's existing identity-view probability or recompute it, with view groups of four or eight. Reuse occurs before view averaging and temperature calibration; row-count checks confirm one actual pass is skipped per full-fallback image. It lowers average validation MACs further to 390,775,315 and preserves 5,718 correct, but neither grouping meets the additional 5% latency reduction requirement against the class-routing reference. Eight-view grouping is not consistently faster. These implementations and negative results are retained, disabled in the selected deployment. No reuse variant receives test evaluation.

    The class-routing recipe was frozen and committed before test evaluation, and was retained after the runtime phase failed its advancement rule. It is this round's only new test endpoint. Test accuracy remains 9,449/10,000 (94.49%), average MACs fall from 475,709,806 to 410,159,652 (13.8%), and parameters remain 3,164,836. Worst-case MACs remain 3,319,207,680. Exactly 3,977 images still exit after the first model; 8,753 now exit before fallback, versus 8,552 previously. The total expensive-fallback count decreases from 1,448 to 1,247. No weights or thresholds change from these test outcomes.
    ''')
    code('''class_recipe=json.loads(Path('results/class_routes_balanced_recipe.json').read_text())
class_record=json.loads(Path('results/class_routes_test_metrics.json').read_text())
if not RETRAIN_ALL:
    class_test=evaluate_refinement('class_routes')
    print('Recorded / executed test correct:',class_record['selected']['balanced']['correct'],class_test['selected']['balanced']['correct'])
else:
    class_test=class_record
    print('Recorded class routing only; retrained weights require validation selection and freeze.')
class_row=class_test['selected']['balanced']
display(pd.DataFrame(class_row['classification_report']).T.round(4))
print('Actual test routing:',class_row['execution'])
display(pd.DataFrame([{'slice':label,'previous_ms':a['median_ms'],'class_routing_ms':b['median_ms'],
                      'time_reduction_%':100*(1-b['median_ms']/a['median_ms'])}
                     for label,a,b in zip(['First 1024','Last 1024'],class_recipe['reference_latency'],class_recipe['timings'])]))
reuse=json.loads(Path('results/reuse_experiment.json').read_text())
assert not reuse['selected']
display(pd.DataFrame([{'candidate':r['label'],'correct':r['correct'],'average_validation_MACs':r['macs'],
                      'first_ms':r['timings'][0]['median_ms'],'last_ms':r['timings'][1]['median_ms'],'passes':r['passes']}
                     for r in reuse['actual_finalists']]))
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].bar(['Previous progressive','Class-conditional'],
            [class_recipe['reference_latency'][0]['median_ms'],class_recipe['timings'][0]['median_ms']])
axes[0].set(ylabel='Median ms / 1,024 validation images',title='Same-session timing; shared RTX 5090')
axes[1].bar(['Previous progressive','Class-conditional'],
            [refined_test['selected']['balanced']['execution']['macs_per_image']/1e6,class_row['execution']['macs_per_image']/1e6])
axes[1].set(ylabel='Average test MACs (millions)',title='Both recipes: 94.49% observed test accuracy')
plt.show()
''')
    md('''Same-session median latency changes from 58.13 to 46.73 ms on the first slice (19.6% lower) and from 56.67 to 41.40 ms on the last slice (27.0% lower). Each contains 1,024 validation images; timing uses three warmups and seven synchronized repetitions, includes views/routing/probabilities/CPU output, and excludes loading/input transfer. Both slices remain development and timing-selection data. These are shared-device throughput results, not independent confirmation or single-image latency guarantees. Validation and the public test benchmark have been repeatedly observed across the project; no class-ranking or significance claim is made.

    Reproduce frozen evaluation with `python3 refine_cascade.py --class-routes --test`. Deploy using `hybrid_experiment.predict` and `results/class_routes_balanced_recipe.json`. The same five checkpoints and all runtime code are embedded. Original reference recipes and test records remain available; the earlier stages in this notebook describe their own historical results.''')
    md('''## 16. Add fallback views only when needed

    The previous reduced-view experiments weakened every hard-image fallback and did not qualify. Here the full 10/30/10-view ensemble remains available, while medium-difficulty images can stop earlier. Keep the class-conditional front, .99 first-pass exit, fallback wide gate, weights, temperatures and four-view grouping unchanged. On reaching the full ensemble, compute **10/10/10 views** first. If its weighted/calibrated maximum probability reaches **0.70**, return that prediction; otherwise append the fine-patch model's remaining twenty views and reconstruct its thirty-view probability before calibration. Other model outputs and the first-ten fine-patch mean are reused. No new model is trained or stored.

    The first ten views exactly match the initial scale-one prefix of the thirty-view definition. Repeating their saved mean ten times in the final average gives the same weight as the original sum in real arithmetic; floating-point rounding and subset batch shape can still change results. Runnable checks verify prefix equivalence within tolerance, actual skipped image-view rows, early/full exits and average/worst-case costs. No compute discount is claimed for work that still executes.

    Compare twelve fixed policies: confidence .70/.80/.90/.95/.975/.99, allowing either all predicted classes or classes 1/5/7/8/9. Cached screening is followed by actual routed recomputation of up to five eligible finalists. Preserve the reference correct count, Shirt F1 and both development halves; add no parameters; require at least 1% fewer average MACs and at least 5% lower latency on both first/last 1,024 validation slices. The .70 all-class rule is selected and committed before test evaluation. Validation remains 5,718/6,000 correct, Shirt F1 0.859829 and halves [2,856,2,862]; average validation MACs fall from 399,102,505 to 324,844,050.

    Test accuracy remains 9,449/10,000 (94.49%), with average MACs falling from 410,159,652 to 339,421,241 (17.2%). Parameters remain 3,164,836; worst-case MACs remain 3,319,207,680. Of 1,112 images reaching three-model inference, 489 exit after the initial views and 623 need the extra twenty fine-patch views. Earlier front and wide-gate route counts are unchanged. No threshold or checkpoint was changed after this test evaluation.
    ''')
    code('''adaptive_recipe=json.loads(Path('results/adaptive_balanced_recipe.json').read_text())
adaptive_record=json.loads(Path('results/adaptive_test_metrics.json').read_text())
if not RETRAIN_ALL:
    adaptive_test=evaluate_refinement('adaptive')
    print('Recorded / executed test correct:',adaptive_record['selected']['balanced']['correct'],adaptive_test['selected']['balanced']['correct'])
else:
    adaptive_test=adaptive_record
    print('Recorded adaptive views only; retrained weights require a new validation freeze.')
adaptive_row=adaptive_test['selected']['balanced']
display(pd.DataFrame(adaptive_row['classification_report']).T.round(4))
print('Actual test routing:',adaptive_row['execution'])
display(pd.DataFrame([{'slice':label,'previous_ms':a['median_ms'],'adaptive_ms':b['median_ms'],
                      'time_reduction_%':100*(1-b['median_ms']/a['median_ms'])}
                     for label,a,b in zip(['First 1024','Last 1024'],adaptive_recipe['reference_latency'],adaptive_recipe['timings'])]))
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].bar(['Fixed fallback views','Adaptive fallback views'],
            [adaptive_recipe['reference_latency'][0]['median_ms'],adaptive_recipe['timings'][0]['median_ms']])
axes[0].set(ylabel='Median ms / 1,024 validation images',title='Same-session timing; shared RTX 5090')
axes[1].bar(['Fixed fallback views','Adaptive fallback views'],
            [class_test['selected']['balanced']['execution']['macs_per_image']/1e6,adaptive_row['execution']['macs_per_image']/1e6])
axes[1].set(ylabel='Average test MACs (millions)',title='Both frozen recipes: 94.49% test accuracy')
plt.show()
''')
    md('''Same-session medians change from 46.66 to 39.64 ms on the first 1,024 validation images (15.1% lower) and from 41.44 to 35.58 ms on the last 1,024 (14.1% lower). Timing includes views, routing, probability aggregation and CPU output, excludes model loading/input transfer, and uses three warmups plus seven synchronized repetitions on the shared RTX 5090. Both slices are reused selection data, not independent speed confirmation or single-image latency measurements. The public test benchmark has been repeatedly observed; unchanged accuracy is not evidence of a generalization improvement.

    Evaluate the frozen option with `python3 refine_cascade.py --adaptive --test`, or call `hybrid_experiment.predict` with `results/adaptive_balanced_recipe.json`. The same five model checkpoints and all runtime code are embedded. Historical reference recipes remain available; optional retraining requires a new validation-only freeze before reusing any adaptive decision policy.''')
    md('''## 17. Automated search in a compact configuration space

    Replace manual trial cycles with a seeded script. Eleven coordinates encode six confidence thresholds (single-model exit, garment/Dress/other agreement exits, wide-model gate and adaptive-view exit), two relative mixture-weight logits and three log-temperature multipliers. The network weights, front's equal weighting, view definitions, model storage and four-view batching stay fixed. This is configuration-space optimization, not a learned neural latent representation or RL.

    Each of sixteen generations proposes 256 candidates: 25% uniform exploration, 25% perturbations around the reference and 50% Gaussian sampling around accumulated elites. Clip proposals to declared bounds, rank feasible candidates by validation correct count then MACs and NLL, and preserve a minimum sampling spread. Seven prediction arrays are cached once using checkpoint, code, split and environment hashes. Candidate scoring broadcasts chunks of 32 across the 6,000 validation examples and accounts for all conditional stages. Completed generations and RNG state support exact-budget resume. No test data is loaded by the search script.

    The 4,096 proposals took **10.37 seconds of cached scoring**, plus 2.31 seconds to construct the prediction cache; these exclude startup, actual finalist checks, timing and packaging. 209 candidates passed cached correct-count, Shirt-F1, both-half, parameter and >=1% MAC-reduction guards. A five-candidate shortlist alternates accuracy-first and compute-first rankings, excluding duplicates of rounded cost/correct-count/half-count signatures. Actual routed inference and >=5% latency reduction on both validation slices are required before selection. All five pass. Candidate 4072 wins by correct count then MACs, with validation 5,720/6,000, halves [2,857, 2,863], unchanged Shirt F1 0.859829 and 287,488,267 average MACs. Cache scoring is an approximation to subset inference; the actual executor supplies final metrics.

    The chosen recipe was committed in `4d9b07e` before test evaluation. Test accuracy is **94.55%**, versus 94.49% for the adaptive reference. Fourteen predictions change: ten errors are fixed and four new errors appear. Average test MACs fall **11.2%**, from 339,421,241 to **301,330,871**. Stored parameters remain 3,164,836 and worst-case MACs remain 3,319,207,680. Test routing uses only model predictions: 4,797 first-pass exits, 8,846 total outer exits, 1,154 fallback images, 146 wide-gate exits, and 506 images requiring the additional fine-patch views.

    More search can overfit a repeatedly reused validation split. The two validation halves and timing slices are development data, not independent confirmation; the public test benchmark has already been observed across this project. Six additional test-correct images do not establish significance or guarantee the class bonus. Shared RTX 5090 timing includes routing, views, softmax and CPU output, excludes loading/input transfer, and uses three warmups and seven synchronized repetitions. No single-image latency claim is made.
    ''')
    code('''search=json.loads(Path('results/latent_search.json').read_text())
latent_recipe=json.loads(Path('results/latent_balanced_recipe.json').read_text())
latent_record=json.loads(Path('results/latent_test_metrics.json').read_text())
from latent_search import self_check as check_cached_search
check_cached_search()
if not RETRAIN_ALL:
    latent_test=evaluate_refinement('latent')
    assert latent_test['selected']['balanced']['correct']==latent_record['selected']['balanced']['correct']
else:
    latent_test=latent_record
    print('Recorded search only; retrained weights require a new validation freeze.')
latent_row=latent_test['selected']['balanced']
print('Candidates / cache-qualified:',search['candidate_count'],search['eligible_count'])
print('Cache construction / candidate scoring seconds:',search['cache_seconds'],search['screening_seconds'])
display(pd.DataFrame(latent_row['classification_report']).T.round(4))
print('Actual test routing:',latent_row['execution'])
display(pd.DataFrame([{'slice':label,'adaptive_ms':a['median_ms'],'search_ms':b['median_ms'],
                      'time_reduction_%':100*(1-b['median_ms']/a['median_ms'])}
                     for label,a,b in zip(['First 1024','Last 1024'],latent_recipe['reference_latency'],latent_recipe['timings'])]))
candidates=pd.read_csv('results/latent_search_candidates.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].scatter(candidates['macs']/1e6,candidates['accuracy']*100,s=5,alpha=.18,color='slateblue')
axes[0].scatter([latent_recipe['macs']/1e6],[latent_recipe['accuracy']*100],s=70,marker='*',color='darkorange',label='Frozen actual endpoint')
axes[0].set(xlabel='Cached average validation MACs (millions)',ylabel='Validation accuracy (%)',title='4,096 proposals; reused development split')
axes[0].legend(fontsize=8)
axes[1].bar(['Adaptive reference','Script-selected'],[adaptive_row['execution']['macs_per_image']/1e6,latent_row['execution']['macs_per_image']/1e6],color=['slategray','darkorange'])
axes[1].set(ylabel='Average test MACs (millions)',title='Frozen test accuracy: 94.49% to 94.55%')
plt.show()
''')
    md('''Use `hybrid_experiment.predict` with `results/latent_balanced_recipe.json` for deployment. `python3 latent_search.py --check` verifies cached probabilities, routed outputs, class metrics and charged MACs. `python3 latent_search.py --resume` runs or resumes the recorded fixed search budget; rerun research in a separate experiment copy because search replaces `results/latent_*`. Default notebook execution only evaluates the frozen endpoint and never repeats candidate selection. All required sources and the unchanged five checkpoints are embedded.''')
    md('## References\n\n'+(ROOT/'SOURCES.md').read_text().split('\n',1)[1])
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':platform_version()}})
    target=ROOT/'submission'/'Assign01_Cai_Haochen_58561440.ipynb'
    target.parent.mkdir(exist_ok=True)
    NotebookClient(notebook,timeout=3600,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(notebook,target)
    number=0
    for cell in notebook.cells:
        for output in cell.get('outputs',[]):
            if 'image/png' in output.get('data',{}):
                number+=1
                (OUT/f'notebook_figure_{number:02d}.png').write_bytes(base64.b64decode(output['data']['image/png']))
    print(target)


def platform_version():
    import platform
    return platform.python_version()


if __name__=='__main__':
    build()
