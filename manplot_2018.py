import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from random import randint
import pyopencl as cl

# Set default context as GPU, code is meant to be run on GPU but CPU will work totally fine
platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
#ctx = cl.create_some_context(interactive=True)     #uncomment this for interactive context

def fractal_gpu(q, maxiter,d1,d2,eqn_list):

    global ctx
    horizon=2.0**15		# For AA, but AA seems to depend on Hausdorff dimension; not working well for random
    log_horizon=np.log(np.log(horizon))/np.log(2)
    queue = cl.CommandQueue(ctx)
    num_eqn = len(eqn_list)
    output = np.empty(q.shape, dtype=np.uint16)

    prg = cl.Program(ctx, """#include <pyopencl-complex.h>
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void fractal(__global float2 *q,
                     __global ushort *output, ushort const maxiter,
                     ushort d1, ushort d2, double horizon, double log_horizon, __global ushort *eqn_list,
		     ushort num_eqn)
    {
        int gid = get_global_id(0);
        cfloat_t c;
        c.real = q[gid].x;
        c.imag = q[gid].y;
        cfloat_t z;
	z.real=0;
	z.imag=0;
        output[gid] = 0;
        for(int curiter = 0; curiter < maxiter; curiter++) {
            if (z.real*z.real + z.imag*z.imag > 16.0f){
            	double az = sqrt(z.real*z.real+z.imag*z.imag);
                output[gid] = curiter;// - log(log(az))/log((float)2)+ log_horizon; // AA - seems dependent on Hausdorff dimension
                return;
            }


            for (int i=0; i < num_eqn; i++){
				if (eqn_list[i] == 0){
					z = cfloat_powr(z,d1);
					z.real = z.real + q[gid].x;
					z.imag = z.imag + q[gid].y;
				} else if (eqn_list[i] == 1) {
					z=cfloat_powr(z,d2);
					z.real = z.real + q[gid].x;
					z.imag = z.imag + q[gid].y;
				} else if (eqn_list[i] == 2){
					z.real = sqrt(z.real*z.real);
					z.imag = sqrt(z.imag*z.imag);
					z = cfloat_powr(z,2);
					z.real = z.real + q[gid].x;
					z.imag = z.imag + q[gid].y;
				} else if (eqn_list[i] == 3) {
					z = cfloat_exp(z);
					z.real = z.real + q[gid].x;
					z.imag = z.imag + q[gid].y;
				} else if (eqn_list[i] == 4) {
					z = cfloat_cos(z);
					z.real = z.real + q[gid].x;
					z.imag = z.imag + q[gid].y;
				} else if (eqn_list[i] == 5) {
					z.real = z.real*z.real - z.imag*z.imag + q[gid].x;
					z.imag = 2*z.real*z.imag + q[gid].y;
				} else if (eqn_list[i] == 6) {
					z.imag = 2*z.real*z.imag + q[gid].y;
					z.real = z.real*z.real - z.imag*z.imag + q[gid].x;
				}
			}
            
        }
    }
    """).build()

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)
    eqn_list = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=eqn_list)\

    prg.fractal(queue, output.shape, None, q_opencl,
                   output_opencl, np.uint16(maxiter), np.uint16(d1), np.uint16(d2),
                   np.float64(horizon),np.float64(log_horizon), eqn_list, np.uint16(num_eqn))
    cl.enqueue_copy(queue, output, output_opencl).wait()
    
    return output



def fractal_set3(xmin,xmax,ymin,ymax,width,height,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    c = np.ravel(c)
    d1=randint(2,4)
    d2=randint(1,3)
    print("d1 = " + str(d1) + "     d2 = " + str(d2))
    num_eqn = randint(2,7)
    eqn_list = np.random.randint(0,7,size=num_eqn)
    print(eqn_list)
    n3 = fractal_gpu(c,maxiter,d1,d2,eqn_list)
    n3 = n3.reshape((width,height))
    return (r1,r2,n3.T)




def fractal_image(xmin,xmax,ymin,ymax,width=10,height=10,maxiter=75,cmap='jet',count=0):
	l = ['viridis','inferno','plasma','magma','Blues','BuGn','BuPu','GnBu','Greens','Oranges',
	     'PuBuGn','PuRd','Purples','RdPu','Reds','YlGn','YlGnBu','YlOrBr','YlOrRd','afmhot',
	     'autumn','cool','copper','gist_heat','hot','pink','spring','summer','winter','BrBG',
	     'bwr','coolwarm','PiYG','PRGn','PuOr','RdBu','RdGy','RdYlBu','RdYlGn','Spectral',
	     'seismic','Dark2','Paired','Pastel1','Pastel2','Set1','Set2','Set3','gist_earth',
	     'terrain','ocean','gist_stern','brg','CMRmap','cubehelix','gnuplot','gnuplot2',
	     'gist_ncar','nipy_spectral','jet','rainbow','gist_rainbow','hsv','flag','prism']
	cmap=l[randint(0,len(l)-1)]
	print("Cmap: " + cmap)
	dpi = 100
	img_width = dpi * width
	img_height = dpi * height
	x,y,z = fractal_set3(xmin,xmax,ymin,ymax,img_width,img_height,maxiter)

	fig, ax = plt.subplots(figsize=(width, height),dpi=dpi)
	ticks = np.arange(0,6*img_width,3*dpi)
	x_ticks = xmin + (xmax-xmin)*ticks/6/img_width
	plt.xticks(ticks, x_ticks)
	y_ticks = ymin + (ymax-ymin)*ticks/6/img_width
	plt.yticks(ticks, y_ticks)
	ax.set_title(cmap)
	norm = colors.PowerNorm(0.3)
	ax.imshow(z.T,cmap=cmap,origin='lower',norm=norm) 
	extent = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
	save_image(fig,extent,count)



def save_image(fig,extent,count):
	global image_counter
	filename = r"C:\Users\Dylan\Desktop\PRSBot\manplotv0_5\FractalPictures\randomfrac_"+str(image_counter)+".png"
	print("Saving Fractal: " + filename)
	image_counter += 1
	fig.savefig(filename,bbox_inches=extent)

#for i in range(200):
#	fractal_image(-2.0,1.0,-1.5,1.5,cmap='gnuplot2')
