def hotmap(df, figsize=(10,8)):
    ##### correlation heatmap
    corr = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.zeros_like(corr, dtype=np.bool)
    idx = np.triu_indices_from(mask)
    mask[idx] = True
    
    sns.heatmap(np.abs(corr),square=True, mask=mask, annot=True,
            cmap=sns.color_palette("magma"),ax=ax,linewidth=2,edgecolor="k")
    ax.set_ylim(len(corr), -.5,.5)
    
    plt.title("CORRELATION BETWEEN VARIABLES")
    plt.show();
    
    ##### descriptive statistics heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(df.describe()[1:].transpose(),annot=True, ax=ax, 
                linecolor="w", linewidth=2,cmap=sns.color_palette("Set2")) #"Set2"
    ax.set_ylim(len(corr), -.5,.5)
    plt.title("Data summary")
    plt.show()
    
    plt.figure(figsize=(13,8))
    
    ### compare proportion of target classes 
    plt.subplot(121)
    ax = sns.countplot(y = df["TARGET"],
                       palette=["b","lime"],
                       linewidth=1,
                       edgecolor="k"*2)
    for i,j in enumerate(df["TARGET"].value_counts().values):
        ax.text(.7,i,j,weight = "bold",fontsize = 27)
    plt.title("Count for target variable in datset")


    plt.subplot(122)
    plt.pie(df["TARGET"].value_counts().values,
            labels=["not pulsars","pulsars"],
            autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
    circ = plt.Circle((0,0),.7,color = "white")
    plt.gca().add_artist(circ)
    plt.subplots_adjust(wspace = .2)
    plt.title("Proportion of target variable in dataset")
    plt.show()

hotmap(df, figsize=(10,8))