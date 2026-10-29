"""3-year backtest of the 4H EMA50/200 cross with an ATR safety stop, across markets.

Stop-and-reverse on the cross (acted on at the next bar open). Optional stop at entry -/+ k x ATR14 (4H);
after a stop, stay flat until the next cross. Reads <market>.csv (datetime, open, high, low, close) from the
current directory. Returns are summed per-trade % (not compounded); usd is P/L on a fixed 0.1 lot.
"""
import pandas as pd,numpy as np
START=pd.Timestamp('2023-10-29')
# $ per 1.0 price move on 0.1 lot (None = % only)
USD={'BTC':lambda e:0.1,'XAU':lambda e:10,'USO':lambda e:61.77,'SLV':lambda e:549.5,'EURUSD':lambda e:10000,'USDJPY':lambda e:10000/e,
     'QQQ':lambda e:None,'SPY':lambda e:None,'DIA':lambda e:None,'ETH':lambda e:None}
def run(df,k,name):
    c,h,l,o=(df[x].values for x in ('close','high','low','open'));t=df.index
    e50=df.close.ewm(span=50,adjust=False).mean().values;e200=df.close.ewm(span=200,adjust=False).mean().values
    tr=np.maximum(h-l,np.maximum(abs(h-np.r_[c[0],c[:-1]]),abs(l-np.r_[c[0],c[:-1]])));atr=pd.Series(tr).rolling(14).mean().values
    out=[];pos=0
    for i in range(201,len(c)):
        tn=1 if e50[i-1]>e200[i-1] else -1;tp=1 if e50[i-2]>e200[i-2] else -1
        if tn!=tp:
            if pos: out.append((t0,pos,ent,o[i]))
            pos=0
            if t[i]>=START: pos=tn;ent=o[i];t0=t[i];stop=ent-pos*k*atr[i-1] if k else None
        if pos and stop is not None and ((pos==1 and l[i]<=stop) or (pos==-1 and h[i]>=stop)):
            px=stop if (pos==1 and o[i]>stop) or (pos==-1 and o[i]<stop) else o[i]
            out.append((t0,pos,ent,px));pos=0
    if pos: out.append((t0,pos,ent,c[-1]))
    r=pd.DataFrame(out,columns=['t','side','ent','ex']);r['pct']=r.side*(r.ex-r.ent)/r.ent*100
    f=USD[name](1);r['usd']=[s*(x-e)*USD[name](e) if f is not None else np.nan for s,e,x in zip(r.side,r.ent,r.ex)]
    return r
def stats(r):
    w=r.pct>0;eq=r.pct.cumsum();dd=(np.maximum.accumulate(np.r_[0,eq])[1:]-eq).max()
    return dict(trades=len(r),win=round(w.mean()*100),net_pct=round(r.pct.sum(),1),pf=round(r.pct[w].sum()/-r.pct[~w].sum(),2) if (~w).any() else 99,
                maxdd_pct=round(dd,1),usd=round(r.usd.sum()) if r.usd.notna().all() else None,
                longs=round(r.pct[r.side==1].sum(),1),shorts=round(r.pct[r.side==-1].sum(),1))
rows=[]
for n in ['BTC','XAU','USO','SLV','QQQ','SPY','DIA','ETH','EURUSD','USDJPY']:
    df=pd.read_csv(n+'.csv',index_col=0,parse_dates=True)
    bh=round((df.close.iloc[-1]/df.close[df.index>=START].iloc[0]-1)*100,1)
    for k in (0,3,4,6):
        rows.append(dict(mkt=n,stop=k,bh=bh,**stats(run(df,k,n))))
R=pd.DataFrame(rows);print(R.to_string(index=False));R.to_csv('results.csv',index=False)
