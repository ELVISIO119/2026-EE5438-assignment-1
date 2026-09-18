"""Evaluate the frozen recipe; never select or modify models from test results."""
import json
import math
import platform
from datetime import datetime,timezone
import numpy as np
import torch
import torchvision
from sklearn.metrics import classification_report,confusion_matrix
from torchvision.datasets import FashionMNIST
from train import OUT,DEVICE
from evaluation import load_model,probabilities,probability_metrics,digest


def evaluate_final():
    recipe=json.loads((OUT/'final_recipe.json').read_text())
    for component in recipe['components']:
        assert digest(component['name'])==component['sha256'], 'Checkpoint changed after recipe freeze.'
    data=FashionMNIST('data',train=False,download=True)
    x=data.data.unsqueeze(1).float().div(255).to(DEVICE)
    y=data.targets
    predictions=[]
    for component in recipe['components']:
        model=load_model(component['name'])
        predictions.append(probabilities(model,x,component['views']))
        del model
    probs=torch.stack(predictions).mean(0)
    pred=probs.argmax(1)
    report=classification_report(y,pred,target_names=data.classes,output_dict=True,zero_division=0)
    scores=probability_metrics(probs,y)
    n=len(y); p=scores['accuracy']; z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    baseline=load_model('baseline_sgd_sigmoid')
    base_probs=probabilities(baseline,x)
    result=dict(**scores,macro_precision=report['macro avg']['precision'],macro_recall=report['macro avg']['recall'],
                macro_f1=report['macro avg']['f1-score'],weighted_f1=report['weighted avg']['f1-score'],
                accuracy_ci95_wilson=[center-half,center+half],classification_report=report,
                confusion_matrix=confusion_matrix(y,pred).tolist(),parameters=recipe['parameters'],macs=recipe['macs'],
                recipe=recipe['name'],recipe_frozen_at=recipe['frozen_at'],evaluated_at=datetime.now(timezone.utc).isoformat(),
                baseline=probability_metrics(base_probs,y),
                baseline_classification_report=classification_report(y,base_probs.argmax(1),target_names=data.classes,output_dict=True,zero_division=0),
                environment=dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,
                                 device=torch.cuda.get_device_name() if DEVICE.type=='cuda' else 'CPU'))
    (OUT/'final_test_metrics.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'final_test_predictions.npz',labels=y.numpy(),probabilities=probs.numpy(),baseline_probabilities=base_probs.numpy())
    print(json.dumps({k:v for k,v in result.items() if k not in ('classification_report','confusion_matrix','baseline_classification_report')},indent=2))
    return result


if __name__=='__main__':
    evaluate_final()
