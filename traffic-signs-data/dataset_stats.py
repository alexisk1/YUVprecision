import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('./train.p', 'rb') as f:
     train = pickle.load(f)

with open('./valid.p', 'rb') as f:
     valid = pickle.load(f)

with open('./test.p', 'rb') as f:
     test = pickle.load(f)

size_tr = train ['sizes']
size_vl = valid ['sizes']
size_te = test ['sizes']

print (size_tr[0][0],"   ",size_tr[0][1])


## Training stats
tr_width =size_tr[:,0]
tr_height =size_tr[:,1]

tr_hist, tr_bin_edges = np.histogram (tr_width,bins=50)
plt.hist(tr_width,bins=tr_bin_edges)
plt.title("Train Width")
plt.show()


## Valid stats
vl_width =size_vl[:,0]
vl_height =size_vl[:,1]

vl_hist, vl_bin_edges = np.histogram (vl_width,bins=50)
plt.hist(vl_width,bins=vl_bin_edges)
plt.title("Valid Width")
plt.show()

## Test stats

te_width =size_te[:,0]
te_height =size_te[:,1]
te_hist, te_bin_edges = np.histogram (te_width,bins=50)
plt.hist(te_width,bins=te_bin_edges)
plt.title("Test Width")
plt.show()


## MERGED
plt.close('all')
width=[]
#width.extend(tr_width)
#width.extend(vl_width)
width.extend(te_width)
width = np.concatenate((tr_width,vl_width,te_width),axis=0)

height=[]
height.extend(tr_height)
height.extend(vl_height)
height.extend(te_height)

mr_hist, mr_bin_edges = np.histogram (width,bins=50)
plt.hist(width,bins=mr_bin_edges)
plt.title("Merge Width",)
plt.draw()

mr_hist, mr_bin_edges = np.histogram (height,bins=50)
plt.hist(height,bins=mr_bin_edges)
plt.title("Merge Width+height")
plt.show()


