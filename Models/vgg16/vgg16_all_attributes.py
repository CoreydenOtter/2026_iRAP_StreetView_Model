"""VGG16 multi-task road attribute baseline. Train on train.jsonl, tune on validation.jsonl, test on test.jsonl."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

DEFAULT_ATTRIBUTES = None  # Automatically select every attribute in the training annotations

def load_rows(path):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            answer=next(m for m in reversed(r['messages']) if m['role']=='assistant')
            content=answer['content']
            raw=''.join(x.get('text','') for x in content if isinstance(x,dict)) if isinstance(content,list) else content
            rows.append((r['image'],json.loads(raw)))
    return rows

def resolve_image(name, root):
    p=Path(name.replace('\\','/'))
    candidates=[Path(root)/p,Path(root)/p.name,Path(root)/'images'/p.name]
    for c in candidates:
        if c.is_file():return c
    raise FileNotFoundError(f'Cannot locate {name}; tried: {candidates}')

class RoadDataset(Dataset):
    def __init__(self,rows,root,attributes,classes,train=False):
        self.rows=rows;self.root=root;self.attributes=attributes;self.classes=classes
        self.transform=transforms.Compose(([transforms.RandomHorizontalFlip()] if train else []) + [transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        path,labels=self.rows[i]
        with Image.open(resolve_image(path,self.root)) as im: x=self.transform(im.convert('RGB'))
        y=[self.classes[a].index(str(labels[a])) if str(labels.get(a)) in self.classes[a] else -100 for a in self.attributes]
        return x,torch.tensor(y,dtype=torch.long)

class VGGMulti(nn.Module):
    def __init__(self,classes):
        super().__init__()
        backbone=models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features=backbone.features
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.heads=nn.ModuleDict({a:nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,len(c))) for a,c in classes.items()})
    def forward(self,x):
        z=self.pool(self.features(x)).flatten(1)
        return {a:head(z) for a,head in self.heads.items()}

def evaluate(model,loader,device,attributes,classes):
    model.eval(); actual={a:[] for a in attributes};pred={a:[] for a in attributes}
    with torch.no_grad():
        for x,y in loader:
            out=model(x.to(device))
            for j,a in enumerate(attributes):
                actual[a].extend(y[:,j].tolist());pred[a].extend(out[a].argmax(1).cpu().tolist())
    results={}
    for a in attributes:
        pairs=[(t,p) for t,p in zip(actual[a],pred[a]) if t!=-100]
        if not pairs:continue
        yt,yp=zip(*pairs)
        mp,mr,mf,_=precision_recall_fscore_support(yt,yp,labels=range(len(classes[a])),average='macro',zero_division=0)
        wp,wr,wf,_=precision_recall_fscore_support(yt,yp,labels=range(len(classes[a])),average='weighted',zero_division=0)
        results[a]={'accuracy':accuracy_score(yt,yp),'macro_precision':mp,'macro_recall':mr,'macro_f1':mf,'weighted_precision':wp,'weighted_recall':wr,'weighted_f1':wf,'support':len(yt)}
    return results,actual,pred

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',default='training_data/train.jsonl');p.add_argument('--val',default='training_data/validation.jsonl');p.add_argument('--test',default='training_data/test.jsonl')
    p.add_argument('--images-root',default='.');p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--attributes',nargs='+',default=DEFAULT_ATTRIBUTES);p.add_argument('--out',default='vgg16_results')
    args=p.parse_args();random.seed(42);np.random.seed(42);torch.manual_seed(42)
    train,val,test=[load_rows(p) for p in (args.train,args.val,args.test)]
    if args.attributes is None:
        args.attributes=sorted({a for _,labels in train for a in labels})
    classes={a:sorted({str(labels[a]) for _,labels in train if labels.get(a) is not None}) for a in args.attributes}
    if any(len(v)<2 for v in classes.values()):raise ValueError(f'Attributes require >=2 training classes: {classes}')
    print(f'Classifying {len(args.attributes)} attributes')
    for name,rows in [('train',train),('validation',val),('test',test)]:
        for path,_ in rows:resolve_image(path,args.images_root)
        print(name,len(rows))
    print('Training classes:',json.dumps(classes,indent=2))
    device='cuda' if torch.cuda.is_available() else 'cpu';print('Device:',device)
    datasets=[RoadDataset(rows,args.images_root,args.attributes,classes,train=i==0) for i,rows in enumerate((train,val,test))]
    loaders=[DataLoader(ds,batch_size=args.batch_size,shuffle=i==0,num_workers=0) for i,ds in enumerate(datasets)]
    model=VGGMulti(classes).to(device)
    for par in model.features.parameters():par.requires_grad=False
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    best=-1
    for epoch in range(args.epochs):
        model.train();total=0
        for x,y in loaders[0]:
            x=x.to(device);y=y.to(device);opt.zero_grad();scores=model(x)
            losses=[nn.functional.cross_entropy(scores[a],y[:,j],ignore_index=-100) for j,a in enumerate(args.attributes) if (y[:,j]!=-100).any()]
            if not losses:continue
            loss=torch.stack(losses).mean();loss.backward();opt.step();total+=loss.item()
        metrics,_,_=evaluate(model,loaders[1],device,args.attributes,classes)
        score=np.mean([v['macro_f1'] for v in metrics.values()])
        print(f'Epoch {epoch+1}/{args.epochs}: training loss sum={total:.3f}, validation mean macro-F1={score:.4f}')
        if score>best:
            best=score;torch.save({'model':model.state_dict(),'classes':classes,'attributes':args.attributes},out/'best_vgg16.pt')
    checkpoint=torch.load(out/'best_vgg16.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model'])
    metrics,actual,pred=evaluate(model,loaders[2],device,args.attributes,classes)
    import csv
    fields=['attribute','accuracy','macro_precision','macro_recall','macro_f1','weighted_precision','weighted_recall','weighted_f1','support']
    with open(out/'attribute_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{'attribute':a,**m} for a,m in metrics.items()])
    with open(out/'predictions.jsonl','w') as f:
        for i,(path,_) in enumerate(test):
            f.write(json.dumps({'image':path,'ground_truth':{a:(classes[a][actual[a][i]] if actual[a][i]!=-100 else None) for a in args.attributes},'prediction':{a:classes[a][pred[a][i]] for a in args.attributes}})+'\n')
    with open(out/'classes.json','w') as f:json.dump(classes,f,indent=2)
    for a,m in metrics.items():print(f'{a}: accuracy={m["accuracy"]:.3f}, macro-F1={m["macro_f1"]:.3f}')
    print('Saved results to',out)
if __name__=='__main__':main()
