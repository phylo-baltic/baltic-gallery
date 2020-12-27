---
title:  "Trait transitions within branches"
metadate: "hide"
categories: [ Phylogeny, Traits ]
image: "/assets/img/reassort.png"
visit: "https://github.com/evogytis/baltic/blob/master/docs/notebooks/austechia.ipynb"
---

# Multitype trees

---

baltic now has the ability to deal with multitype trees recovered as part of structured coalescent analyses, which contain nodes with a single child. You can find an example of the files you might find after running a structured coalescent analysis in beast2 [here](https://github.com/Taming-the-BEAST/Structured-coalescent/).

In this particular example, it is possible to plot a tree and colour the different types along the branches to show trait transitions.

First, fetch the tree as a nexus file and load it with baltic.

```python
address='https://raw.githubusercontent.com/Taming-the-BEAST/Structured-coalescent/master/precooked_runs/MTT.h3n2_2deme.map.trees' ## address of example tree
fetch_tree = requests.get(address) ## fetch tree
treeFile=sio(fetch_tree.text) ## stream from repo copy

mtt=bt.loadNexus(treeFile,absoluteTime=False) ## treeFile here can alternatively be a path to a local file
mtt.treeStats() ## report stats about tree
mtt.sortBranches()
```

Then , create the plot. Parts of the tree where the trait state is Hong Kong are red, anything else is blue.

```python
fig,ax = plt.subplots(figsize=(10,10),facecolor='w')

c_func=lambda k: 'indianred' if k.traits['type']=='HongKong' else 'steelblue'
n_target=lambda k: k.branchType=='node' and len(k.children)==1

mtt.plotTree(ax,colour=c_func)

kwargs={'marker':'|','lw':2}
mtt.plotPoints(ax,target=n_target,size=30,colour=c_func,zorder=101,**kwargs)

mtt.plotPoints(ax,size=30,colour=c_func,zorder=100)

ax.set_ylim(-1,mtt.ySpan+1)

[ax.spines[loc].set_visible(False) for loc in ['top','right','left','bottom']] ## no axes

ax.tick_params(axis='x',size=0) ## no labels
ax.tick_params(axis='y',size=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.show()
```

{% jupyter_notebook "https://github.com/evogytis/baltic/blob/master/docs/notebooks/austechia.ipynb" %}

---

Attribution-ShareAlike 4.0 International
