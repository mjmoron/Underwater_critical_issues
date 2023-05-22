#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:25:17 2022

@author: alberto
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.close('all')


datos = pd.read_excel('(UUV) TablaResumen_Consolidada_NoR_NUEVA.xlsx')


'Países'

pais = datos.iloc[3:,8].dropna()

pais = pais.str.split(pat=', ', expand=True) # elimino espacios

pais = pais.stack()

pais = pais.str.split(pat=',', expand=True)  # separo todos los países

pais = pais.stack()

pais=pais.replace('Spain ','Spain')
pais=pais.replace('South Korea','Korea')
pais=pais.replace('Saudi Arabia','S. Arabia')
pais=pais.replace('Czech Republic','Czech R.')



num = np.unique(pais.values, return_counts=True)

df = pd.DataFrame([num[0], num[1]]).T

df = df.sort_values(by=1, ascending=False)

df.index=df.iloc[:,0]

df = df.iloc[:,1]




'Citas'

citas = datos.iloc[3:123,7:11]

citas.drop(citas.columns[[1,2]], axis=1, inplace=True)

citas = citas.replace(np.nan, 0).astype(int)

# citas = citas.replace(0, np.nan).dropna(how='all', axis=0)

anno = datos.iloc[3:123, 7].dropna()

df2 = citas

df2 = df2.sort_values(by='Año', ascending=True)

df2 = df2.replace(df2.iloc[:,1].max(), np.nan).dropna() # limpieza valor extremo


periodo = np.unique(df2.iloc[:,0].values)

suma_periodo = []

for i in periodo:
    
    suma = int( df2.where(df2.iloc[:,0]== i ).dropna().sum()[1] )
    
    suma_periodo.append(suma) 


df2_plot = pd.DataFrame([periodo, suma_periodo]).T

df2_plot.index = df2_plot.iloc[:,0]




'Algoritmos por periodo años'


alg = datos.iloc[3:123, 17:23].replace(np.nan, 0).astype(int)

alg.drop(alg.columns[[1,4]], axis=1, inplace=True)

df3 = pd.concat([alg, anno], axis=1).dropna().astype(int)


df3_anual = df3.groupby('Año').sum()

aux = df3_anual

aux['intervalos'] = np.arange(len(aux))

gr1 = aux.iloc[:6,:-1].sum()

gr2 = aux.iloc[6:10,:-1].sum()

gr3 = aux.iloc[10:15,:-1].sum()

gr4 = aux.iloc[15:,:-1].sum()


gr = pd.concat([gr1,gr2,gr3,gr4], axis=1)

gr.columns = ['[1990-2005]','[2006-2010]','[2011-2015]','[2016-2022]']

gr.index = ['Classical algorithms','ANN & NSC','ML Regression & SVM','OBIA']




'Gráfico circular plataforma procesamiento'

cat = datos.iloc[3:123, 29:34].replace(np.nan, 0)#.astype(int)

cat.drop(cat.columns[[2]], axis=1, inplace=True)

onboard = cat.where(cat.iloc[:,0]==1).dropna()
onboard.drop(onboard.columns[[0,1]], axis=1, inplace=True)

offboard = cat.where(cat.iloc[:,1]==1).dropna()
offboard.drop(offboard.columns[[0,1]], axis=1, inplace=True)


aux = onboard.stack().replace(0,np.nan).dropna().sort_values()
aux2 = offboard.stack().replace(0,np.nan).dropna().sort_values()

aux_cont = np.unique(aux.values, return_counts=True)
aux2_cont = np.unique(aux2.values, return_counts=True)


on = pd.DataFrame([aux_cont[0], aux_cont[1]]).T

on.index = aux.index.get_level_values(1)

off = pd.DataFrame([aux2_cont[0], aux2_cont[1]]).T

off['order'] = np.array([ 7,  4,  2,  1,  3,  6, 10, 11,  9,  8,  5])

on = on.sort_values(by=1)
off = off.sort_values(by='order')


onboard_bool = onboard.where(onboard.values==0, 1).sum()

offboard_bool = offboard.where(offboard.values==0, 1).sum()

onoff = pd.DataFrame([onboard_bool.values, offboard_bool.values])

onoff.index = ['Onboard processing', 'Offline processing']

onoff.columns = ['CPU', 'GPU']








'GRÁFICA PAÍSES'

# df[::-1].plot(kind='barh', colormap='viridis_r')

sns.barplot(df.index, df.values, palette="viridis_r")

# sns.barplot(df.values, df.index, palette="viridis_r")
# plt.gca().get_yaxis().set_visible(False)

plt.title('Publications by country',size=16)

# plt.xlabel('',size=10)

plt.xticks(rotation=90)

plt.ylabel('Publications count',size=14)

plt.legend().remove()

plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})



'GRÁFICA CITAS'

plt.figure()
# df2_plot.iloc[:,-1].plot(kind='bar', color=plt.cm.Paired(range(len(df2_plot))))

sns.barplot(df2_plot.index, df2_plot.iloc[:,-1].values, palette="crest_r")


# plt.title('Cites per publications date',size=16)

plt.xlabel('',size=10)

plt.xticks(rotation=90)

plt.ylabel('Cites',size=14)

plt.legend().remove()

# plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})





'GRÁFICA ALGORITMOS POR PERIODO AÑOS'



# sns.barplot(gr.index, gr.iloc[:,-1], palette="magma")

gr['percentage'] = gr.sum(axis=1)[:]*100/gr.sum(axis=1).sum()

ax = gr.iloc[:,:-1].plot(kind='bar', stacked=True, colormap='flare_r') #flare_r, crest_r



for container in ax.containers[::-1]: # le doy la vuelta pq empieza a numerar abajo
    
    ax.bar_label(container, labels=['36%','33%','9%','22%'])
    break


# plt.title('Underwater image processing algorithms',size=16)

# plt.xlabel(list(gr.index),size=10)

plt.xticks(rotation=10)

plt.ylabel('Publications count',size=14)

# plt.legend(bbox_to_anchor=(1,0), ncol=len(gr.columns),fontsize=12)



'GRÁFICA ALGORITMOS PORCENTAJE TOTAL'

plt.figure()

index=['Classical','ANN & NSC','ML Regression & SVM','OBIA']
colores = ['lightblue','orange','green','red']

df_alg = pd.DataFrame(alg.sum().values,index=index)

df_alg['percentage'] = df_alg.iloc[:,0].values*100/df_alg.sum().values

# df_alg.plot(kind='bar')

# plt.bar(df_alg.T.index, alg.sum().values, color=colores)

ax = sns.barplot(data=df_alg, x=df_alg.index, y='percentage')

for container in ax.containers:
    ax.bar_label(container, fmt='%.0f%%')

# for container in [22.1,9.5,32.6,35.8]:
#     ax.bar_label(container, fmt='%.1f%%')


plt.xticks(rotation=10)

# plt.gca().get_xaxis().set_visible(False)

plt.title('Global proportion of underwater imaging algorithms',size=16)

plt.ylabel('Count',size=14)

# plt.legend(bbox_to_anchor=(1,0), ncol=len(df_alg.columns),fontsize=12)
# plt.legend(index)
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})





'GRÁFICA CIRCULAR: PROCESSING PLATFORM'

plt.figure()
plt.pie(off.iloc[:,1], labels= off.iloc[:,0], textprops={'fontsize': 12},\
        autopct='%.1f%%')#, explode=(0.1,0,0,0,0,0,0,0,0,0,0), shadow=True)

# plt.title('Underwater offline CPU/GPU processing',size=16)


plt.figure()
plt.pie(on.iloc[:,1], labels= on.iloc[:,0])

# plt.title('Underwater onboard CPU/GPU processing',size=16)


plt.figure()

plt.rcParams['text.color'] = 'black'

labs = ['Onboard processing (CPU)','Onboard processing (GPU)',\
        'Offline processing (CPU)','Offline processing (GPU)']

plt.pie(onoff.stack().values, labels= labs, textprops={'fontsize': 12},\
        autopct='%.1f%%')#, explode=(0.1,0.1,0,0), shadow=True)

# plt.title('Computing framework',size=16)


# inner_circle = plt.Circle( (0,0), 0.7, color='white')
# p = plt.gcf()
# p.gca().add_artist(inner_circle)



# YlGnBu


'GRÁFICA BARRAS HORIZONTAL: PAPERS POR INTERVALOS TIEMPO'


lord = np.sort(datos.iloc[3:123,7].astype(int).values)

# Búsqueda de intervalos

g = np.unique(lord, return_counts=True)

data = np.array([g[0],g[1]]).T

g1=len(np.where(data[:,0]<=2005)[0])

g2=len(np.where(data[g1:len(data),0]<=2010)[0])

g3=len(np.where(data[g1+g2:len(data),0]<=2015)[0])

g4=len(data[g1+g2+g3:len(data),0])


# Sumas de intervalos

s1 = data[:g1 ].sum(axis=0)[1]

s2 = data[g1:g1+g2 ].sum(axis=0)[1]

s3 = data[g1+g2:g1+g2+g3 ].sum(axis=0)[1]

s4 = data[g1+g2+g3:].sum(axis=0)[1]


x = np.array([2005,2010,2015,2022])
xlabel = ['[1990-2005]','[2006-2010]','[2011-2015]','[2016-2022]']

y = np.array([s1,s2,s3,s4])


df4 = pd.DataFrame([s1,s2,s3,s4]).T
df4.columns = xlabel



# HORIZONTAL BAR PLOT

# df.plot(kind='barh', stacked=True)
plt.figure()

plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})


colors =  {"[2016-2022]":"tab:red",
           "[2011-2015]":"tab:green", 
           "[2006-2010]":"tab:orange",
           '[1990-2005]':'tab:blue'}

sns.barplot(data=df4.T[::-1].T, orient='h', saturation=.9, palette = colors)



# plt.gca().get_yaxis().set_visible(False)

# plt.title('Publications count',size=16)

# plt.ylabel(size=14)

plt.xlabel('Cumulative sum of articles',size=14)
















