

# Example Plots
# https://exoplanetarchive.ipac.caltech.edu/exoplanetplots/

# LINEPLOTS
compare = df.groupby('TARGET')[['MEAN_IP', 'STD_IP', 'KURTOSIS_IP', 'SKEWNESS_IP',
                                        'MEAN_CURVE', 'STD_CURVE', 'KURTOSIS_CURVE',
                                        'SKEWNESS_CURVE']].mean().reset_index()


compare = compare.drop('TARGET', axis=1)

# compare mean of target class varibales
compare_mean = compare.transpose().reset_index()
compare_mean = compare_mean.rename(columns={'index':"features", 0:"not_pulsar", 1:"pulsar"})
plt.figure(figsize=(13,14))
plt.subplot(211)
sns.pointplot(x="features",y="not_pulsar",data=compare_mean,color="b")
sns.pointplot(x="features",y="pulsar",data=compare_mean,color="lime")
plt.xticks(rotation=45)
plt.xlabel("")
plt.grid(True,alpha=.3)
plt.title("COMPARING MEAN OF ATTRIBUTES FOR TARGET CLASSES")

# compare standard deviation of target class variables
compare1 = df.groupby('TARGET')[['MEAN_IP', 'STD_IP', 'KURTOSIS_IP', 'SKEWNESS_IP',
                                        'MEAN_CURVE', 'STD_CURVE', 'KURTOSIS_CURVE',
                                        'SKEWNESS_CURVE']].std().reset_index()
compare1 = compare1.drop('TARGET',axis=1)


compare_std = compare1.transpose().reset_index()
compare_std = compare_std.rename(columns={'index':"features", 0:"not_pulsar", 1:"pulsar"})
plt.subplot(212)
sns.pointplot(x="features",y="not_pulsar",data=compare_std,color="b")
sns.pointplot(x="features",y="pulsar",data=compare_std,color="lime")
plt.xticks(rotation=45)
plt.grid(True,alpha=.3)
plt.title("COMPARING STANDARD DEVIATION OF ATTRIBUTES FOR TARGET CLASSES")
plt.subplots_adjust(hspace =.4)
print ("[GREEN == PULSAR , BLUE == NON-PULSAR]")
plt.show()


# DISTRIBUTION
import itertools
columns = ['MEAN_IP', 'STD_IP', 'KURTOSIS_IP', 'SKEWNESS_IP',
           'MEAN_CURVE', 'STD_CURVE', 'KURTOSIS_CURVE','SKEWNESS_CURVE']
length  = len(columns)
colors  = ["r","lime","b","m","orangered","c","k","orange"] 

plt.figure(figsize=(13,20))
for i,j,k in itertools.zip_longest(columns,range(length),colors):
    plt.subplot(length/2,length/4,j+1)
    sns.distplot(df[i],color=k)
    plt.title(i)
    plt.subplots_adjust(hspace = .3)
    plt.axvline(df[i].mean(),color = "k",linestyle="dashed",label="MEAN")
    plt.axvline(df[i].std(),color = "b",linestyle="dotted",label="STANDARD DEVIATION")
    plt.legend(loc="upper right")
    
print ("***************************************")
print ("DISTIBUTION OF VARIABLES IN DATA SET")
print ("***************************************")




sns.pairplot(df,hue="TARGET")
plt.title("pair plot for variables")
plt.show()


# SCATTERPLOTS
plt.figure(figsize=(14,7))

##### FIRST PLOT
plt.subplot(121)
plt.scatter(x='KURTOSIS_IP',y='SKEWNESS_IP', data=df[df['TARGET'] == 1],alpha=.7,
            label="PULSARS", s=30, color='cyan',linewidths=.4,edgecolors="black")
plt.scatter(x='KURTOSIS_IP',y='SKEWNESS_IP', data=df[df['TARGET'] == 0],alpha=.6,
            label="NOT PULSARS",s=30,color ="b",linewidths=.4,edgecolors="black")
## VLINES
plt.axvline(df[df['TARGET'] == 1]['KURTOSIS_IP'].mean(),
            color = "k",linestyle="dashed",label='PULSAR Mean')
plt.axvline(df[df['TARGET'] == 0]['KURTOSIS_IP'].mean(),
            color = "magenta",linestyle="dashed",label ='NON-PULSAR Mean')
## HLINES
plt.axhline(df[df['TARGET'] == 1]['SKEWNESS_IP'].mean(),
            color = "k",linestyle="dashed")
plt.axhline(df[df['TARGET'] == 0]['SKEWNESS_IP'].mean(),
            color = "magenta",linestyle="dashed")
## LABELS
plt.legend(loc='best')
plt.xlabel("Kurtosis Integrated Profile")
plt.ylabel("Skewness Integrated Profile")
# plt.title("Scatter plot for skewness and kurtosis for target classes")

##### SECOND PLOT
plt.subplot(122)
plt.scatter(x='SKEWNESS_CURVE',y='KURTOSIS_CURVE',data=df[df['TARGET'] == 0],alpha=.7,
            label='NOT PULSARS',s=30,color ="blue",linewidths=.4,edgecolors="black")
plt.scatter(x='SKEWNESS_CURVE',y='KURTOSIS_CURVE',data=df[df['TARGET'] == 1],alpha=.7,
            label="PULSARS",s=30,color = "cyan",linewidths=.4,edgecolors="black")
## VLINES
plt.axvline(df[df['TARGET'] == 1]['KURTOSIS_CURVE'].mean(),
            color = "k",linestyle="dashed",label ="PULSAR Mean")
plt.axvline(df[df['TARGET'] == 0]['KURTOSIS_CURVE'].mean(),
            color = "magenta",linestyle="dashed",label ="NON-PULSAR Mean")
## HLINES
plt.axhline(df[df['TARGET'] == 1]['SKEWNESS_CURVE'].mean(),
            color = "k",linestyle="dashed")
plt.axhline(df[df['TARGET'] == 0]['SKEWNESS_CURVE'].mean(),
            color = "magenta",linestyle="dashed")
## LABELS
plt.legend(loc ="best")
plt.xlabel("Skewness DM-SNR Curve")
plt.ylabel("Kurtosis DM-SNR Curve")
plt.title("Scatter plot for skewness and kurtosis of dmsnr_curve for target classes")
plt.subplots_adjust(wspace =.4)





# BOXPLOTS
columns = [x for x in df.columns if x not in ['TARGET']]
length  = len(columns)
plt.figure(figsize=(13,20))
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot(4,2,j+1)
    sns.lvplot(x=df['TARGET'],y=df[i],palette=["blue","cyan"])
    plt.title(i)
    plt.subplots_adjust(hspace=.3)
    plt.axhline(df[i].mean(),linestyle = "dashed",color ="k",
                label ="Mean value for data")
    plt.legend(loc="best")
    
print ("****************************************************")
print ("BOXPLOT FOR VARIABLES IN DATA SET WITH TARGET CLASS")
print ("****************************************************")






# STACKPLOTS
st = df[df['TARGET'] == 1].reset_index()
nst= df[df['TARGET'] == 0].reset_index()
new = pd.concat([nst,st]).reset_index()

plt.figure(figsize=(13,10))
plt.stackplot(new.index,new['MEAN_IP'],
              alpha =.5,color="b",labels=['MEAN_IP'])
plt.stackplot(new.index,new['STD_IP'],
              alpha=.7,color="c",labels=['STD_IP'])
plt.stackplot(new.index,new['SKEWNESS_IP'],
              alpha=.5,color ="orangered",labels=['SKEWNESS_IP'])
plt.stackplot(new.index,new['KURTOSIS_IP'],
              alpha=.8,color = "magenta",labels=['KURTOSIS_IP'])

plt.axvline(x=16259,color = "black",linestyle="dashed",
            label = "PULSARS vs NON-PULSARS")
plt.axhline(new['MEAN_IP'].mean(),color = "b",
            linestyle="dashed",label = "Average Mean Profile")
plt.axhline(new['STD_IP'].mean(),color = "c",
            linestyle="dashed",label = "Average Std Profile")
plt.axhline(new['SKEWNESS_IP'].mean(),color = "orangered",
            linestyle="dashed",label = "Average Skewness Profile")
plt.axhline(new['KURTOSIS_IP'].mean(),color = "magenta",
            linestyle="dashed",label = "Average Kurtosis Profile")
plt.legend(loc="best")
plt.title("Area plot for attributes for pulsar stars vs non pulsar stars")
plt.show()





### CHANGE TO PLOTLY GRAPHY
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(13,13))
ax  = fig.add_subplot(111,projection = "3d")






ax.scatter(df[df["TARGET"] == 1][["MEAN_IP"]],
           df[df["TARGET"] == 1][["STD_IP"]],
           df[df["TARGET"] == 1][["SKEWNESS_CURVE"]],
           alpha=.5, s=80, linewidth=2, edgecolor="w",
           color="lime", label="Pulsar")

ax.scatter(df[df["TARGET"] == 0][["MEAN_IP"]],
           df[df["TARGET"] == 0][["STD_IP"]],
           df[df["TARGET"] == 0][["SKEWNESS_CURVE"]],
           alpha=.5, s=80, linewidth=2, edgecolor="w",
           color="b", label="Non-Pulsar")

ax.set_xlabel("MEAN_IP", fontsize=15)
ax.set_ylabel("STD_IP", fontsize=15)
ax.set_zlabel("SKEWNESS_CURVE",fontsize=15)
plt.legend(loc="best")
fig.set_facecolor("w")
plt.title("MEAN_PROFILE VS STD_PROFILE VS SKEWNESS_DMSNR_CURVE",
          fontsize=10)
plt.show()

sns.jointplot(df['MEAN_IP'],df['STD_IP'],kind="kde",scale=10)
plt.show()

columns = [x for x in df.columns if x not in ['TARGET']]
length  = len(columns)

plt.figure(figsize=(13,25))

for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot(length/2,length/4,j+1)
    sns.violinplot(x=df['TARGET'],y=df[i],
                   palette=["blue","cyan"],alpha=.5)
    plt.title(i)



    # BARPLOTS
columns = [x for x in df.columns if x not in ['TARGET']]
length  = len(columns)

plt.figure(figsize=(13,25))

for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot(length/2,length/4,j+1)
    sns.barplot(x=df['TARGET'],y=df[i],
                   palette=["blue","lime"],alpha=.7)
    plt.title(i)




### Feature by Feature Comparison Scatterplots

f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)

sns.scatterplot(x='MEAN_IP', y='KURTOSIS_IP',
                hue='TARGET', size='SKEWNESS_IP',
                palette=['b','c'],
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)



f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)

sns.scatterplot(x='SKEWNESS_CURVE', y='KURTOSIS_IP',
                hue='TARGET', size='MEAN_CURVE',
                palette=['b','c'],
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)

sns.scatterplot(x='KURTOSIS_IP', y='KURTOSIS_CURVE',
                hue='TARGET', size='MEAN_CURVE',
                palette=['b','c'],
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)

sns.scatterplot(x='KURTOSIS_IP', y='STD_IP',
                hue='TARGET', size='KURTOSIS_IP',
                palette=['b','c'],
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)


f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)

sns.scatterplot(x='KURTOSIS_IP', y='KURTOSIS_CURVE',
                hue='TARGET', size='KURTOSIS_IP',
                palette=['b','c'],
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)