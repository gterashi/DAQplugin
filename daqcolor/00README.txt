#Uninstall plugin
devel clean ./daqcolor

#Install plugin
devel install ./daqcolor

#Show help page
help daqcolor

daqcolor apply npyPath model [k an integer] [colormap a colormap] [metric a text string] [atomName a text string] [clampMin a number] [clampMax a number]
    — Color residues once from a numpy (N×32) probability file. For metric, you can use aa_score - DAQ(AA) scoreatom_score - DAQ(CA) scoreaa_conf:[AA type] - DAQ(Selected AA type) score
  npyPath: a text string

daqcolor clear
    — Close all marker models created by 'daqcolor points'

daqcolor monitor model [npyPath a text string] [k an integer] [colormap a colormap] [metric a text string] [atomName a text string] [on true or false]
    — Start/stop live recoloring (new frame trigger)

daqcolor points npyPath [radius a number] [metric a text string] [colormap a colormap] [clampMin a number] [clampMax a number]
    — Show xyz points from a numpy file as markers
  npyPath: a text string

#In commandline window
##Colored by aa_score
daqcolor apply ./points_AA_ATOM_SS_swap.npy #2 metric aa_score  k 1
##Colored by atom_score
daqcolor apply ./points_AA_ATOM_SS_swap.npy #1 metric atom_score  k 1


#Save PDB file with window averaging
save colored.pdb #2
