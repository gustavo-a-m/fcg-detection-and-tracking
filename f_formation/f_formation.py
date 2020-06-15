import numpy as np
from scipy import io

# gt= np.array([[0, 0, 1, 2], [0, 0, 1, 2]])

# f= np.array([[[0, -7.7975397, 18.184526, -1.105197],
#     [1, -158.2054, 273.60474, 0.25927138],
#     [2, 78.95437, 62.456608, 31184.951],
#     [5, 82.695786, 847.7799, 1.0792824]]])
# print(f)

class FFormation:
    def __init__(self, f, stride=35):
        self.f = np.array([f])
        self.xrange = range;
        self.est = self.make_est(self.f, stride, 3000)

    def vis(self, title=False):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider #, Button, RadioButtons
        from matplotlib.patches import Ellipse
        gt= np.array([[0, 0, 1, 2], [0, 0, 1, 2]])

        fig = plt.figure()
        if title :
            fig.suptitle(title, fontsize=14, fontweight='bold')

        ax = plt.subplot(111, aspect='equal')
        fig.subplots_adjust(left=0.25, bottom=0.25)
        ax.clear()

        axcolor = 'lightgoldenrodyellow'
        ax2 = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        

        def update(val):  
            val=int(val)
            ax.clear()
            loc=self.find_locs(self.f[val])
            if self.est is not False:
                p=np.random.permutation(self.est[val].max()+1)
                # p2=np.random.permutation(gt[val].max()+1)
                self.calc_distance_vis(loc,self.f[val],p[self.est[val]],3500,ax)
            ax.scatter(self.f[val][:,1],self.f[val][:,2], vmin=0, s=100)
        update(0)
        # slider = Slider(ax2, 'Frame', 0, gt.shape[0] - 1,
        #                 valinit=0, valfmt='%i')

        # slider.on_changed(update)

        plt.show()

    def calc_distance_vis(self, loc,f,labels,mdl,ax):
        from matplotlib.patches import Ellipse
        ax.scatter(f[:,1],f[:,2], c=labels,vmin=0, vmax=labels.max(),s=400)
        ax.plot(np.vstack((f[:,1],loc[:,0])),
                np.vstack((f[:,2],loc[:,1])),'g')
                    
        u=np.unique(labels)
        dist=np.empty((loc.shape[0],u.shape[0]))
        dist2=np.zeros_like(dist)
        for i in self.xrange(u.shape[0]):
            means=loc[labels==i,:].mean(0)
            ells = Ellipse(means,np.sqrt(mdl), np.sqrt(mdl),0)
            ells.set_alpha(0.1)
            ax.add_artist(ells)
            
            dist[:,i]=((loc-means)**2).sum(1)
            mask=np.arange(loc.shape[0])[dist[:,i]<mdl]
            #means=means.T
            disp=f[:,1:3].copy()
            disp-=means
            for j in mask:
                for k in mask:
                    distk=np.linalg.norm(disp[k])
                    distj=np.linalg.norm(disp[j]) 
                    if distk>distj:
                        inner=disp[k].dot(disp[j])
                        norm=distk*distj
                        if inner/norm>.5:
                            print (j,k,disp[j],disp[k])
                            print (distk,distj,inner,norm,distk/distj)
                            dist2[k,i]+=10**(distk/distj)
                            ax.plot(np.vstack((disp[k,0]+means[0],means[0])),
                                    np.vstack((disp[k,1]+means[1],means[1])),'r')
                            ax.plot(np.vstack((disp[j,0]+means[0],means[0])),
                                    np.vstack((disp[j,1]+means[1],means[1])),'b')    
            dist+=dist2
        return dist

    def find_locs(self, f, stride=35):
        "Estimate focal centers for each person given features"
        locs=np.empty((f.shape[0],2))
        locs[:,0]=f[:,1]+np.cos(f[:,3])*stride
        locs[:,1]=f[:,2]+np.sin(f[:,3])*stride
        return locs
        
    def calc_distance_old(self, loc,labels,mdl):
        u=np.unique(labels)
        dist=np.empty((loc.shape[0],u.shape[0]))
        dist2=np.zeros_like(dist)
        for i in self.xrange(u.shape[0]):
            means=loc[labels==i,:].mean(0)
            disp=loc-means
            dist[:,i]=(disp**2).sum(1)
            mask=np.arange(loc.shape[0])[dist[:,i]<mdl]
            for j in mask:
                for k in mask:
                    if dist[k,i]>dist[j,i]:
                        inner=disp[k].dot(disp[j])
                        norm=np.sqrt(dist[k,i]*dist[j,i])
                        if inner/norm>.9:
                            dist2[k,i]+=100**(dist[k,i]/dist[j,i])
            dist+=dist2
        return dist

    def calc_distance(self, loc,f,labels,mdl):
        """Given focal localtions, raw locations(f) and initial labelling l find
        cost of assigning  people to new locations given by the mean of their 
        labelling"""           
        u=np.unique(labels)
        dist=np.empty((loc.shape[0],u.shape[0]))
        for i in self.xrange(u.shape[0]):
            means=loc[labels==i,:].mean(0)      
            dist[:,i]=((loc-means)**2).sum(1)
            #computed sum-squares distance, now
            mask=np.arange(loc.shape[0])[dist[:,i]<mdl]
            disp=f[:,1:3].copy()
            disp-=means
            for j in mask:
                for k in mask:
                    distk=np.linalg.norm(disp[k])
                    distj=np.linalg.norm(disp[j]) 
                    if distk>distj:
                        inner=disp[k].dot(disp[j])
                        norm=distk*distj
                        if inner/norm>.75:
                            dist[k,i]+=100**(inner/norm*distk/distj)
        return dist

    def init(self, locs,f,mdl):
        return self.calc_distance(locs,f,np.arange(locs.shape[0]),mdl)

    def gc(self, f,stride=35,MDL=3500):
        """Runs graphcuts"""
        locs=self.find_locs(f,stride)
        unary=self.init(locs,f,MDL)
        seg=np.full(f.shape[0], fill_value=-1, dtype=np.double)
        
        aux = 0
        for i, cost in enumerate(unary):
            for j in range(i, len(unary)):
                if cost[j] < MDL:
                    if seg[i] == -1:
                        seg[i] = aux
                        aux = aux + 1
                    seg[j] = seg[i]
            if seg[i] == -1:
                seg[i] = aux
                aux = aux + 1

        seg=seg.astype(np.int)
        return seg

    def make_est(self, f, stride=35, mdl=3500):
        """Solve entire sequence"""
        self.est=np.empty(f.shape[0],dtype=object)
        for i in self.xrange(f.shape[0]):
            self.est[i]=self.gc(f[i],stride,mdl)
        return self.est