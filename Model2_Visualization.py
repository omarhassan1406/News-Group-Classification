import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot  as plt

prediction  = np.load('Predictions.npy', allow_pickle=True)
Y_test = np.load('E:/NLP_dataset/Test_labels.npy', allow_pickle=True)
cm = confusion_matrix(Y_test , prediction , labels =  ['alt.atheism', 'comp.graphics' ,'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware' ,'comp.windows.x',
 'misc.forsale', 'rec.autos' ,'rec.motorcycles' ,'rec.sport.baseball',
 'rec.sport.hockey' ,'sci.crypt' ,'sci.electronics' ,'sci.med' ,'sci.space',
 'soc.religion.christian' ,'talk.politics.guns' ,'talk.politics.mideast',
 'talk.politics.misc' ,'talk.religion.misc'])
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm , display_labels=['alt.atheism','comp.graphics','comp.os.ms-windows.misc',
          'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x',
          'misc.forsale','rec.autos','rec.motorcycles',
          'rec.sport.baseball','rec.sport.hockey','sci.crypt',
          'sci.electronics','sci.med','sci.space',
          'soc.religion.christian','talk.politics.guns','talk.politics.mideast',
          'talk.politics.misc','talk.religion.misc'])

cm_display.plot()
plt.show()