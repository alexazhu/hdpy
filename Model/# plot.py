# plot

minus_log10pvalues = -np.log10(pvalues)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #colors = ['red','green','blue', 'yellow']
    x_labels = []
    x_labels_pos = []
    group.plot(kind='scatter', x='ind', y='minuslog10pvalue',color=colors[num % len(colors)], ax=ax)
    x_labels.append(name)
    x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([0, len(df)])
    ax.set_ylim([0, 3.5])
    ax.set_xlabel('Chromosome')