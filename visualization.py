import matplotlib.pyplot as plt
import numpy as np
def plot_curve(x,y,x_t=None,y_t=None,title=None,filename='test.png'):
	plot=plt.plot(x,y)
	fig, ax = plt.subplots() 
	ax.plot(x,y)
	if x_t!=None:
		ax.plot(x_t,y_t)
	if title!=None:
		plt.title(title)
	ax.autoscale()
	fig.savefig(filename)



def test_matplotlib():
	circle1=plt.Circle((0, 0), 0.2, color='r')
	circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
	circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)
	fig, ax = plt.subplots() 
	ax.add_artist(circle1)
	ax.add_artist(circle2)
	ax.add_artist(circle3)
	fig.savefig('plotcircles.png')


def test_draw_circle():
	t = np.linspace(0,2*np.pi*3,1000)
	r=np.linspace(0,1,num=300)
	while r.shape[0]<1000:
		r=np.append(r,1)
	x=r*np.sin(t)
	y=r*np.cos(t)
	plot_curve(x,y)



# test_draw_circle()