# 5y backtest: current rule (hold until cross / ATR stop) vs fixed 1:2 target on fresh cross
import pandas as pd,numpy as np,sys,json,io,glob
START=pd.Timestamp('2021-10-30')
B3='../bt3/';D='/home/claude/kronos/martingale/data/'
K={'BTC':3,'ETH':3,'XAU':3,'QQQ':3,'USO':4,'PPLT':4,'SPY':4,'SLV':6}
USD={'BTC':.1,'ETH':.1,'XAU':10,'USO':61.77,'SLV':549.5,'PPLT':1130,'QQQ':4.09,'SPY':1.003}
def raw(f): return pd.read_csv(io.StringIO(json.load(open(f))['result']),sep=';',parse_dates=['datetime']).set_index('datetime')[['open','high','low','close']].astype(float)
def load(n):
    if n in('USO','SLV'): return pd.read_csv(D+f'{n}_4h.csv.gz',parse_dates=['datetime']).set_index('datetime')[['open','high','low','close']]
    if n in('BTC','ETH','XAU'):
        a=raw(f'raw/{n}_a.json');b=pd.read_csv(B3+n+'.csv',index_col=0,parse_dates=True)
        return pd.concat([a,b[b.index>a.index[-1]]]).sort_index()
    return raw(f'raw/{n}.json').sort_index()
def run(df,k,mode,R=2.0):
    """mode 'hold': exit on opposite cross or stop. mode 'rr': also take profit at entry+side*R*risk."""
    c,h,l,o=(df[x].values for x in('close','high','low','open'));t=df.index
    e50=df.close.ewm(span=50,adjust=False).mean().values;e200=df.close.ewm(span=200,adjust=False).mean().values
    pc=np.r_[c[0],c[:-1]];tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc)));atr=pd.Series(tr).rolling(14).mean().values
    out=[];pos=0
    for i in range(201,len(c)):
        tn=1 if e50[i-1]>e200[i-1] else -1;tp=1 if e50[i-2]>e200[i-2] else -1
        if tn!=tp:
            if pos: out.append((t0,pos,ent,o[i],'cross'))
            pos=0
            if t[i]>=START:
                pos=tn;ent=o[i];t0=t[i];risk=k*atr[i-1];stop=ent-pos*risk;tgt=ent+pos*R*risk
        if not pos: continue
        hitS=(pos==1 and l[i]<=stop) or (pos==-1 and h[i]>=stop)
        hitT=mode=='rr' and ((pos==1 and h[i]>=tgt) or (pos==-1 and l[i]<=tgt))
        if hitS:  # stop first if both in one bar (conservative)
            px=stop if (pos==1 and o[i]>stop) or (pos==-1 and o[i]<stop) else o[i]
            out.append((t0,pos,ent,px,'stop'));pos=0
        elif hitT:
            px=tgt if (pos==1 and o[i]<tgt) or (pos==-1 and o[i]>tgt) else o[i]
            out.append((t0,pos,ent,px,'target'));pos=0
    if pos: out.append((t0,pos,ent,c[-1],'open'))
    return pd.DataFrame(out,columns=['t','side','ent','ex','why'])
def stats(r,usd):
    r=r.copy();r['pct']=r.side*(r.ex-r.ent)/r.ent*100;r['usd']=r.side*(r.ex-r.ent)*usd if usd else np.nan
    w=r.pct>0;eq=r.pct.cumsum();dd=(np.maximum.accumulate(np.r_[0,eq])[1:]-eq).max()
    return dict(trades=len(r),win=round(w.mean()*100) if len(r) else 0,net_pct=round(r.pct.sum(),1),maxdd_pct=round(dd,1) if len(r) else 0,
                usd=round(r.usd.sum()) if usd else None,tgt=(r.why=='target').sum(),stp=(r.why=='stop').sum())
if __name__=='__main__':
    rows=[]
    for n in sys.argv[1:]:
        df=load(n);k=K.get(n,3);usd=USD.get(n)
        assert df.index[0]<START-pd.Timedelta(days=30),(n,df.index[0])
        bh=round((df.close.iloc[-1]/df.close[df.index>=START].iloc[0]-1)*100,1)
        for mode in('hold','rr'):
            rows.append(dict(mkt=n,k=k,mode=mode,bh=bh,first=str(df.index[0])[:10],**stats(run(df,k,mode),usd)))
    R=pd.DataFrame(rows);print(R.to_string(index=False));R.to_csv('res_'+'_'.join(sys.argv[1:3])+'.csv',index=False)
