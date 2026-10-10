import sys, numpy as np, pandas as pd
sys.path.insert(0,'sim')
from generator import Config, Generator
r2=float(sys.argv[1]) if len(sys.argv)>1 else 0.8
g=Generator(Config(lam=1.0,r2_phq=r2,r2_gad=r2))
X=g.sample(200000,seed=1); df=pd.DataFrame(X)
cols=['age','education_years','monthly_income','comorbidities','pa_minutes_week','sleep_hours','phq9','gad7']
print(df[cols].corr('spearman').round(2))
print(df[['age','education_years','monthly_income','comorbidities','sleep_hours','pa_days_week','pa_minutes_session','pa_minutes_week','phq9','gad7']].describe().T[['mean','std','min','max']].round(2))
print('PHQ classes',np.round(np.bincount(df.phq_class)/len(df),3),' GAD',np.round(np.bincount(df.gad_class)/len(df),3))
print('female',df.gender_female.mean().round(3),'marital',np.round(np.bincount(df.marital_status)[1:]/len(df),3),'living',np.round(np.bincount(df.living_situation)[1:]/len(df),3))
print('pa_type',np.round(np.bincount(df.pa_type)/len(df),3),'intensity',np.round(np.bincount(df.pa_intensity)/len(df),3))
P=g.class_proba(X,'phq'); print('Bayes acc PHQ',np.mean(P.argmax(1)==df.phq_class).round(4), 'mean max p',P.max(1).mean().round(4))
P=g.class_proba(X,'gad'); print('Bayes acc GAD',np.mean(P.argmax(1)==df.gad_class).round(4))
print('coef',{k:{t:round(c,3) for t,c in v.items()} for k,v in g.coef.items()})
