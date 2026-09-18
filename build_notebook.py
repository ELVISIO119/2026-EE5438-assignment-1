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
    recipe=json.loads((OUT/'final_recipe.json').read_text())
    metrics=json.loads((OUT/'final_test_metrics.json').read_text())
    evidence={p.stem:json.loads(p.read_text()) for p in OUT.glob('*.json')}
    class_metrics={name:row for name,row in metrics['classification_report'].items() if isinstance(row,dict) and 'avg' not in name}
    weakest=min(class_metrics,key=lambda name:class_metrics[name]['f1-score'])
    classes=list(class_metrics)
    confusions=sorted([(count,classes[i],classes[j]) for i,row in enumerate(metrics['confusion_matrix']) for j,count in enumerate(row) if i!=j],reverse=True)[:3]
    files=[ROOT/name for name in ('models.py','train.py','baseline.py','sharpness.py','prune.py',
                                  'evaluation.py','select_final.py','evaluate_final.py','SOURCES.md','requirements.txt')]
    files+=list((ROOT/'configs').glob('*.json'))+list(OUT.glob('*.json'))+list(OUT.glob('*.csv'))
    files += [OUT/f'{name}.pt' for name in sorted({'baseline_sgd_sigmoid'}|{c['name'] for c in recipe['components']})]
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

    The frozen final classifier obtains **{metrics['accuracy']:.2%} test accuracy**, macro precision **{metrics['macro_precision']:.4f}**, macro recall **{metrics['macro_recall']:.4f}**, and macro F1 **{metrics['macro_f1']:.4f}** on 10,000 Fashion-MNIST test images. All components are MLPs. No convolution, attention, Transformer, outside training images or pretrained weights are used.

    The technical summary and limitations at the end explain which changes helped, which did not, and the full inference cost. Accuracy meets the assignment's highest published accuracy threshold when at least 93%; marks and top-ten bonus are determined by the instructor, not guaranteed by this notebook.
    ''')
    md('''## 1. Reproducibility and data discipline

    **Run All performs real evaluation of the frozen weights.** The portable bundle below contains readable Python sources, configuration files, recorded training histories, and the selected checkpoints. It is unpacked into a new temporary working directory, leaving existing files untouched. The bundle is not a remote dependency.

    Set `RETRAIN_ALL=True` to rerun the full training sequence from scratch before validation selection and test evaluation. This is deliberately opt-in because it includes all negative-result experiments. Recorded training histories are labelled as recorded evidence; they are not represented as newly executed training in the default evaluation run. PyTorch 2.10, torchvision 0.25, NumPy, pandas, scikit-learn and matplotlib are required. CUDA is recommended; CPU evaluation is supported but slower.

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
ax.set_title('Pure MLP inference: dense token/channel mixing and probability aggregation')
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
    for name in ['accuracy_p2','accuracy_wide','accuracy_muon_refine','accuracy_residual']:
        run(json.loads(Path(f'configs/{name}.json').read_text()))
    from select_final import select
    select(accuracy_first=True)
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
                 'epochs':cfg.get('epochs',50),'best_epoch':r.get('best_epoch'),
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
    code('''pareto=pd.read_csv('results/pareto.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,cost in zip(axes,['parameters','macs']):
    ax.scatter(pareto[cost],100*pareto.val_accuracy,c=pareto.pareto.astype(int),cmap='coolwarm',s=38)
    for _,row in pareto[pareto.pareto].iterrows():
        ax.annotate(row['name'],(row[cost],100*row.val_accuracy),fontsize=7,xytext=(4,4),textcoords='offset points')
    ax.set_xscale('log'); ax.set_xlabel(cost); ax.set_ylabel('Validation accuracy (%)'); ax.grid(alpha=.2)
plt.tight_layout(); plt.show()
display(pareto[pareto.pareto].sort_values('macs'))
''')
    md('Pareto dominance is defined jointly over validation accuracy (higher), physical parameters (lower) and dense MACs (lower). A model with fewer parameters is not necessarily cheaper: Mixer weights are reused over tokens. FLOPs are approximately twice dense MACs, excluding nonlinearities, normalization, memory traffic and augmentation. This chart describes single-model, single-view checkpoints, not ensemble cost.')
    md('''## 4. Frozen final evaluation

    The accuracy-first search evaluates 1/2/4/10/18/30/50-view inference for completed models within 3.5 percentage points of the best raw validation score. The ten best distinct checkpoints contribute equal-weight pairs, triples and prefixes. Up to three starting models also undergo eight greedy convex-weight additions, with fixed candidate weights 0.1/0.2/0.35/0.5. Each accepted greedy step must avoid reducing correct count on either fixed 3,000-image validation half. Both halves remain development data; this guard is not an independent test or a significance claim. All attempted weight settings are counted in `weighted_search.json`, including rejected ones.

    Ten views consist of the image and four one-pixel cardinal translations, each with and without horizontal flip. Eighteen views use every 3x3 translation offset with both flip states. Thirty views combine the ten-view setting with affine-grid scales 1/0.96/1.04, using bilinear interpolation; fifty views cover every 5x5 translation offset with both flip states. Padding is zero, never wraparound. The earlier 1/2/4-view definitions remain unchanged. Selection maximizes total validation correct count, with validation negative log-likelihood as its only tie-breaker. Parameters and MACs are recorded but have no selection penalty or cutoff. The fixed candidate search is exploratory and cannot guarantee a global optimum or the highest class ranking.
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
fig,axes=plt.subplots(2,6,figsize=(14,5))
for ax,idx in zip(axes.flat,errors):
    ax.imshow(test_data.data[idx],cmap='gray'); ax.axis('off')
    ax.set_title(f'True: {test_data.classes[labels[idx]]}\\nPred: {test_data.classes[predicted[idx]]}',fontsize=8)
plt.suptitle('First 12 errors in test-index order (not cherry-picked)'); plt.tight_layout(); plt.show()
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

    Enlarging the inference and ensemble search lets models combine complementary errors. The selected component table records each model, number of views, exact weight when nonuniform, and checkpoint hash. The final selection rule uses validation correct count and NLL only; training and inference costs remain visible for assessment, even though they are no longer optimization constraints.

    ### Regularization and model efficiency

    The 2x2 control yields neither dropout nor decay {acc('regularization_none')}, dropout only {acc('regularization_dropout')}, decay only {acc('regularization_decay')}, and both {acc('residual_gelu')}. These results describe one fixed architecture and seed; they do not establish a universally optimal regularizer.

    Retaining 75% of channel neurons gives {acc('mixer_pruned_75_initial')} before recovery and {acc('mixer_pruned_75')} after fine-tuning; retaining 50% gives {acc('mixer_pruned_50_initial')} and {acc('mixer_pruned_50')}. The compression removes real matrix dimensions and its fine-tuning cost is additional. The Pareto table reports measured accuracy against both parameter and computation cost. Initial pruning damage and recovered accuracy are reported separately.

    ### Final result and cost

    The frozen recipe is **{recipe['name']}**. It obtains **{metrics['correct']}/10,000 correct ({metrics['accuracy']:.2%})** on the official test split, compared with **{metrics['baseline']['accuracy']:.2%}** for the fixed sigmoid/SGD reference. Macro precision is **{metrics['macro_precision']:.4f}**, macro recall **{metrics['macro_recall']:.4f}**, macro F1 **{metrics['macro_f1']:.4f}**, and weighted F1 **{metrics['weighted_f1']:.4f}**. The confusion matrix and per-class table identify remaining class-specific errors.

    Deployment requires **{recipe['parameters']:,} total stored parameters** and **{recipe['macs']:,} dense MACs per image**, approximately **{2*recipe['macs']:,} dense FLOPs**. These totals include all ensemble members and every inference view. They must not be presented as the cost of one single-pass model. Averaged weights themselves add no deployment model beyond the resulting checkpoint.

    The validation candidate table retains alternatives and their cost. No independent test evaluation of every candidate is used to choose the displayed winner. A higher validation score from a large search may reflect both genuine error complementarity and validation overfitting; small differences should not be interpreted as proven generalization improvements.

    ### Limits and next steps

    All experiments use one assignment-specific seed and one validation split. Repeated validation selection may overfit that split; architecture comparisons with different budgets and regularizers are recipe comparisons, not clean causal ablations. Shared-device wall time is descriptive, not a controlled hardware benchmark. The descriptive Wilson 95% interval for final test accuracy is [{metrics['accuracy_ci95_wilson'][0]:.2%}, {metrics['accuracy_ci95_wilson'][1]:.2%}]; it does not account for adaptive validation search, prior public-benchmark exposure, dataset shift, or prove that small differences are significant. This is an exploratory benchmark continuation, not a newly blinded evaluation. Further research should use independent seeds and a new validation protocol, rather than tuning against this test score.

    Section A is supplied separately as a typed study guide. The assignment requires the student's genuine handwritten or iPad-written answers in one PDF. A typed guide is not a compliant handwritten submission. Review and understand the answers and code, and follow the course's assistance/disclosure rules before submission.
    ''')
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
