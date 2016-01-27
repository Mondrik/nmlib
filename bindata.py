import numpy as np

def bindata(x, y, nbins=10, binsize=None):
    #put in functionality for specifying bin size instead of 
    #no. of bins eventually
    binsize = (np.max(x)-np.min(x))/nbins
    #there will be nbins+1 bin edges (since we place an edge at max(x)
    #but only nbins centers, as one would expect
    bin_edges = np.min(x) + np.array(range(nbins+1))*binsize
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    contents = [[] for i in range(nbins)]
    for j,val in enumerate(x):
        for i in range(nbins):
            if val >= bin_edges[i] and val < bin_edges[i+1]:
                contents[i].append(y[j])
                break
    return bin_centers, contents
