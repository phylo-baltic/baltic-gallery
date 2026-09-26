import baltic as bt
from baltic.curonia import get_variable_aln_sites, plot_snp_alignment

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

def custom_selection_fxn(columnDict, validNucleotideFxn, refNt):
    """
    Takes a dict of {seq name: nt} of individual alignment columns, a function to identify valid nucleotides (by default A, C, T, U, G, -) and reference nucleotide at site returns Boolean of whether to keep alignment column.
    """
    from collections import Counter
    # focalSeqs = ['OR278367.1|SARS-CoV-2/human/FRA/IHUCOVID-052808/2021', 
    #              'OX000736.1|IMSSC2-206-2021-00058', 
    #              'OR272792.1|SARS-CoV-2/human/FRA/IHUCOVID-010977/2021', 
    #              'OR278345.1|SARS-CoV-2/human/FRA/IHUCOVID-052814/2021', 
    #              'OR275459.1|SARS-CoV-2/human/FRA/IHUCOVID-014681/2021', 
    #              'OR322339.1|SARS-CoV-2/human/FRA/IHUCOVID-011654/2021']

    focalSeqs = ['PP846209.1|SARS-CoV-2/human/FRA/IHUCOVID-102487/2022', 
                 'PP846150.1|SARS-CoV-2/human/FRA/IHUCOVID-102344/2022', 
                 'PP847473.1|SARS-CoV-2/human/FRA/IHUCOVID-101421/2022', 
                 'PP846149.1|SARS-CoV-2/human/FRA/IHUCOVID-102343/2022']
    keep = False
    clean_column = [columnDict[seq] for seq in columnDict if validNucleotideFxn(columnDict[seq])] ## filter to valid nucleotides
    focal_column = [columnDict[seq] for seq in focalSeqs if validNucleotideFxn(columnDict[seq])] ## get variable sites within focal set
    count_focal_nts = Counter(focal_column)
    
    siteVariable = len(set(clean_column)) >= 2 ## column contains more than one unique value
    allFocalDifferent = focal_column.count(refNt) == 0 ## all focal sequences different at site
    focalVariable = Counter(focal_column).most_common()[1][1] >= 2 if len(set(focal_column)) >= 2 else False ## 2nd most common variable site in focal sequences found in at least 2 sequences
    
    if siteVariable and (allFocalDifferent or focalVariable): ## site variable AND (all focal sequences differ from reference or at least 2 focal sequences have a different nt at site)
        keep = True

    return keep

##############
fig = plt.figure(figsize=(20,10),facecolor='w')
gs = GridSpec(1, 2, width_ratios = [1, 10], hspace = 0.01, wspace = 0.0)
ax = plt.subplot(gs[1])
ax2 = plt.subplot(gs[0])

treePath = 'SARS-CoV-2.aln.newick'

ll = bt.io.load_newick(treePath, 'divergence')
ll.sort_branches(descending=False)

## choice of reference - sequence in alignment, consensus of alignment or external file
# refSeq = 'OR322336.1|SARS-CoV-2/human/FRA/IHUCOVID-017931/2021'
# refSeq = 'consensus'
refSeq = 'NC_045512.gb'

alnPath = 'SARS-CoV-2.aln.fasta'

## get indices of variable sites
SNPs = get_variable_aln_sites(alnPath, trimStart=150, trimEnd=150, refSeq=refSeq, refSeqFmt='gb', selectionFxn=custom_selection_fxn)

gffFile = 'annotations.gff'

## plot SNP alignment
plot_snp_alignment(alnAx=ax, treeAx=ax2, SNPs=SNPs, alnFile=alnPath, tree=ll, refSeq=refSeq, coding=True, gffFile=gffFile, refSeqFmt='gb', plotORFs=True, offsetORFs=0.2)

ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.0005))
plt.savefig('snp-alignment.png', bbox_inches='tight')