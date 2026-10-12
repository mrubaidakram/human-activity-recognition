import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from .data import ensure_data, load_split
from .model import ActivityMLP

LABELS = ("walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying")
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def evaluate(model, loader, device):
    model.eval(); actual=[]; predicted=[]
    with torch.inference_mode():
        for x,y in loader:
            output = model(x.to(device)).argmax(1).cpu().numpy(); predicted.extend(output.tolist()); actual.extend(y.numpy().tolist())
    return {"accuracy": accuracy_score(actual, predicted), "macro_f1": f1_score(actual, predicted, average="macro"),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=range(6)).tolist(),
        "classification_report": classification_report(actual, predicted, labels=range(6), target_names=LABELS, output_dict=True, zero_division=0)}
def loader(x,y,batch_size,shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle, drop_last=False)
def main():
    parser=argparse.ArgumentParser(description="Train a subject-separated PyTorch MLP on UCI HAR.")
    parser.add_argument("--data-dir",default="data"); parser.add_argument("--output-dir",default="artifacts")
    parser.add_argument("--epochs",type=int,default=40); parser.add_argument("--batch-size",type=int,default=128); parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1: parser.error("epochs and batch-size must be positive")
    seed_all(args.seed); root=ensure_data(args.data_dir); xtrain,ytrain,subjects=load_split(root,"train"); xtest,ytest,_=load_split(root,"test")
    train_ids,val_ids=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=args.seed).split(xtrain,ytrain,groups=subjects))
    scaler=StandardScaler().fit(xtrain[train_ids]); xfit=scaler.transform(xtrain[train_ids]).astype(np.float32); xval=scaler.transform(xtrain[val_ids]).astype(np.float32); xtest=scaler.transform(xtest).astype(np.float32)
    fit_loader=loader(xfit,ytrain[train_ids],args.batch_size,True); val_loader=loader(xval,ytrain[val_ids],args.batch_size,False); test_loader=loader(xtest,ytest,args.batch_size,False)
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"); model=ActivityMLP(xtrain.shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss(); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); best=-1
    with (output/"metrics.jsonl").open("w",encoding="utf-8") as log:
        for epoch in range(1,args.epochs+1):
            model.train(); loss_total=rows=0
            for x,y in fit_loader:
                x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(x),y); loss.backward(); optimizer.step(); loss_total+=loss.item()*len(y); rows+=len(y)
            val=evaluate(model,val_loader,device); record={"epoch":epoch,"train_loss":loss_total/rows,"validation":val}; print(json.dumps(record)); log.write(json.dumps(record)+"\n"); log.flush()
            if val["macro_f1"]>best:
                best=val["macro_f1"]; torch.save({"model_state":model.cpu().state_dict(),"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"labels":LABELS,"epoch":epoch},output/"best.pt"); model.to(device)
    saved=torch.load(output/"best.pt",map_location=device,weights_only=True); model.load_state_dict(saved["model_state"]); result={"selected_epoch":saved["epoch"],"test":evaluate(model,test_loader,device)}
    (output/"test_metrics.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result))
if __name__=="__main__": main()
