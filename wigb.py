import numpy as np
import matplotlib.pyplot as plt

def wigb(data=None,scale=None,x=None,z=None, ax=None,amx=None):
#  WIGB: Plot seismic data using wiggles.
#
#  WIGB(data,scale,x,z,amx) 
#
#  IN  data:     地震数据 (data ndarray, traces are columns)
#      scale: multiple data by scale
#      x:     x轴 (often offset)
#      z:     y轴 (often time)
#      amx:  最大振幅
#
#  Note
#
#    If only 'data' is enter, 'scale,x,z,amn,amx' are set automatically; 
#    otherwise, 'scale' is data scalar; 'x, z' are vectors for annotation in 
#    offset and time, amx are the amplitude range.
#
    if data is None:
        nx, nz = 10, 10
        data = np.random.random((nz,nx))
        # print(data)
    nz, nx = data.shape

    trmx = np.max(np.abs(data),axis=0)
    if amx is None:
        amx = np.mean(trmx)
    if x is None:
        x = np.arange(1,nx+1,1)
    if z is None:
        z = np.arange(1,nz+1,1)
    if scale is None:
        scale = 1
    
    if nx <=1:
        print('ERR:PlotWig: nx has to be more than 1')
        return 
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # take the average as dx
    dx1 = np.abs(x[1:nx]-x[0:nx-1])
    dx = np.median(dx1)
    dz = z[1]-z[0]
    xmax, xmin = data.max(), data.min()

    data = data * dx / amx
    data = data * scale

    print(' PlotWig: data range [%f, %f], plotted max %f \n'%(xmin,xmax,amx))

    # set display range
    x_min = min(x) - 2.0*dx         # 左边界
    x_max = max(x) - 2.0*dx
    z_min = min(z) - dz             # 下边界
    z_max = max(z) - dz
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([z_min,z_max])

    zstart, zend = z[0], z[-1]      # 上下边界

    fillcolor = [0, 0, 0]
    linecolor = [0, 0, 0]
    linewidth = 1.


    for i in range(0,nx):           # 遍历每一道
        if not trmx[i]==0:
            tr = data[:,i]          # x=i 一道数据
            s = np.sign(tr)         # 一道数据的符号
            i1 = []
            for j in range(0,nz):   # 遍历每一个时间点
                if j==nz-1:
                    continue
                if not s[j]==s[j+1]:
                    i1.append(j)
            npos = len(i1)

            i1 = np.array(i1)       # 一道数据的振幅变化点
            if len(i1)==0:
                zadd=np.array(())
            else:
                zadd = i1 + tr[i1] / (tr[i1]-tr[i1+1])
            aadd = np.zeros(zadd.shape)  # 一道数据的振幅变化点的振幅

            zpos = np.where(tr>0)       # 一道数据的正振幅点
            tmp = np.append(zpos,zadd)  # 一道数据的正振幅点和振幅变化点
            zz = np.sort(tmp)           # 一道数据的正振幅点和振幅变化点排序

            iz = np.argsort(tmp)        # 一道数据的正振幅点和振幅变化点排序的索引
            aa = np.append(tr[zpos],aadd)   # 一道数据的正振幅点和振幅变化点的振幅
            iz=iz.astype(int)        # 一道数据的正振幅点和振幅变化点排序的索引
            aa = aa[iz]             # 一道数据的正振幅点和振幅变化点的振幅排序


            if tr[0]>0:
                a0,z0=0,1.00
            else:
                a0,z0=0,zadd[0]
            if tr[nz-1]>0:
                a1,z1=0,nz
            else:
                a1,z1=0,zadd.max()

            zz = np.append(np.append(np.append(z0,zz),z1),z0)
            aa = np.append(np.append(np.append(a0,aa),a1),a0)

            zzz = zstart + zz * dz - dz         # 一道数据的正振幅点和振幅变化点的时间

            ax.fill(aa+x[i], zzz+dz, color=fillcolor)
            ax.plot(x[i]+[0,0],[zstart,zend],color=[1,1,1])
            ax.plot(x[i]+tr,z,color=linecolor,linewidth=linewidth)

        else:
            ax.plot([x[i],x[i]],[zstart,zend],color=linecolor,linewidth=linewidth)


    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')  # 将x轴的标签也移动到顶部

    ax.set_xlabel('Offset (m)')
    ax.set_ylabel('Time (s)')
    
    
    plt.show()


