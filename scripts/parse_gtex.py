#/usr/bin/python

import numpy as np 
import cmapPy as cmap 
import os

# dictionary for each sample type

tissues = {
	"Adipose-Subcutaneous" : 350,
	"Adipose-Visceral" : 227,
	"Adrenal-Gland" : 145,
	"Artery-Aorta" : 224,
	"Artery-Coronary" : 133,
	"Artery-Tibial" : 332,
	"Bladder" : 11,
	"Brain-Amygdala" : 72,
	"Brain-Anterior" : 84,
	"Brain-Caudate" : 117,
	"Brain-Cerebellar-Hemisphere" : 105,
	"Brain-Cerebellum" : 125,
	"Brain-Cortex" : 114,
	"Brain-Frontal" : 108,
	"Brain-Hippocampus" : 94,
	"Brain-Hypothalamus" : 96,
	"Brain-Nucleus" : 113,
	"Brain-Putamen" : 97,
	"Brain-Spinal" : 71,
	"Brain-Substantia" : 63,
	"Breast-Mammary" : 214,
	"Cells-EBV" : 118,
	"Cells-Transformed" : 284,
	"Cervix-Ectocervix" : 6,
	"Cervix-Endocervix" : 5,
	"Colon-Sigmoid" : 149,
	"Colon-Transverse" : 196,
	"Esophagus-Gastroesophageal" : 153,
	"Esophagus-Mucosa" : 286,
	"Esophagus-Muscularis" : 247,
	"Fallopian Tube" : 6,
	"Heart-Atrial" : 194,
	"Heart-Left" : 218,
	"Kidney-Cortex" : 32,
	"Liver" : 119,
	"Lung" : 320,
	"Minor-Salivary-Gland" : 57,
	"Muscle-Skeletal" : 430,
	"Nerve-Tibial" : 304,
	"Ovary" : 97,
	"Pancreas" : 171,
	"Pituitary" : 103,
	"Prostate" : 106,
	"Skin-Not-Sun-Exposed" : 250,
	"Skin-Sun-Exposed" : 357,
	"Small-Intestine" : 88,
	"Spleen" : 104,
	"Stomach" : 193,
	"Testis" : 172,
	"Thyroid" : 323,
	"Uterus" : 83,
	"Vagina" : 96,
	"Whole-Blood" : 393
}

#read GTEx RNA-seq data into GCToo object
if os.path.isfile('gtex.npy'):
	gtex_data = np.load('gtex.npy')
	cols = np.load('cols_gtex.npy')
else:
	print('\nExtracting data from GCT file...\n')
	myGC = cmap.pandasGEXpress.parse("GTEx_Analysis_v6p_RNA-seq_RNA-SeQCv1.1.8_gene_rpkm.gct")
	#save GTEx data as numpy ndarray
	gtex_data = myGC.data_df.as_matrix()
	#save column headers
	cols = np.array(myGC.col_metadata_df.index.values)
	np.save('gtex.npy', gtex_data)
	np.save('cols_gtex.npy', cols)

j = 0

#loop through each example and save it to a h5 file
for key in sorted(tissues.iterkeys()):
	print('creating files for ' + key)
	#create class directory if not already there
	if not os.path.exists('GTEx_Data/' + key):
		os.makedirs('GTEx_Data/' + key)

	#create h5 file for each sample
	for i in range(tissues[key]):
		gtex_data[:,j].tofile('GTEx_Data/' + key + '/' + str('%03d' % i) + '_' + cols[j] + '.dat', sep="")
		# np.savetxt('GTEx_Data/' + key + '/' + str('%03d' % i) + '_' + cols[j] + '.txt', gtex_data[:,j], fmt='%8f')
		j = j + 1

print('\nDone!\n')

