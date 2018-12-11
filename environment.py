import rospy
from random import randint
from utils import *
import numpy as np
from ROSNode import ROSNode



class Env:
	def __init__(self,height):
		self.ROSNode=None
		self.target_pos=None
		self.height=height

		self.limitbox=[-20,20,-20,20,height-2,height+2]
		self.n_observation=12
		self.n_action=2
		self.interact_rate=20
		
		self.local_position = None

	
		self.state_buf2=[0]*self.n_observation
		self.last_move=[0]*self.n_action #object's move
		self.rl_last_act=[0,0] #network's last action
		self.last_control_err4=[0,0]
		self.last_control_err3=[0,0]
		self.last_control_err2=[0,0]
		self.last_control_err=[0,0]
		self.cur_control_err=[0,0]
		self.t_step=0
		
	def wait(self):
		self.ROSNode.hover(1)

	def curve(self,tstep):
		# if self.env is not None:
		# 	rate=self.env.ROSNode.rate
		# else:
		# 	rate=10
		# t=1.0*tstep/rate
		# r=4
		# if t<20:
		# 	r=r*(t/20.0)
		# x=r*np.cos(0.5*t)
		# y=r*np.sin(0.5*t)

		rate=self.interact_rate

		t=1.0*tstep/rate
		r=4
		if t<20:
			r=r*(t/20.0)

		# x=r*np.cos(0.5*t)+0.5*np.cos(2*t);
		# y=r*np.sin(0.5*t)+0.5*np.cos(0.5*t);	
		# x=r*np.cos(0.5*t);
		# y=r*np.sin(0.5*t);	
		# if abs(y)>abs(x):
		# 	y=r*y/abs(y)
		# else:
		# 	x=r*x/abs(x)
		den=abs(np.sin(0.5*t))+abs(np.cos(0.5*t))
		x=r*np.sin(0.5*t)/den
		y=r*np.cos(0.5*t)/den

		return x,y
	#reset all environment
	def reset(self):
		self.ROSNode=ROSNode()
		self.ROSNode.start()
		self.wait()
		self.target_pos=[0,0,self.height]
		# s,_=self.get_state()
		self.t_step=0
		
		x,y,z,qx,qy,qz,qw=self.ROSNode.get_uav_pos()
		tx,ty=self.curve(0.0)
		self.state_buf2=[0]*self.n_observation

		return np.reshape(np.array(self.state_buf2),[-1])






	#test+visual	
	def get_gaussian_score(self,diff):
		score=0
		sigma_p2=3
		# attenuation=0.8
		x1,y1,z1=0,0,0
		x2,y2,z2=diff[0],diff[1],diff[2]
	
		dis=get_dis(x1,y1,z1,x2,y2,z2)
		score+=10*np.exp(-( dis**2 / ( sigma_p2 ) ) )
		score-=0.8

		return score



	def step(self,act):
		self.t_step+=1
		self.target_pos[0],self.target_pos[1]=self.curve(self.t_step)
		self.ROSNode.publish_target_point(self.target_pos[0],self.target_pos[1],self.target_pos[2])

		# print(act)
		tx=act[0]*3+self.target_pos[0]
		ty=act[1]*3+self.target_pos[1]
		tz=self.target_pos[2]
		self.ROSNode.moveto(tx,ty,tz)
		self.ROSNode.ros_interact_step(1.0/self.interact_rate)
		x,y,z,qx,qy,qz,qw=self.ROSNode.get_uav_pos()


	

		r=self.get_gaussian_score((self.target_pos[0]-x,self.target_pos[1]-y,0))
		done=False

		if x<self.limitbox[0] or x>self.limitbox[1] or y<self.limitbox[2] or y>self.limitbox[3] or z<self.limitbox[4] or z>self.limitbox[5]:
			done=True

		if get_dis(x,y,z,tx,ty,tz)>10:
			done=True
		self.last_control_err4=self.last_control_err3
		self.last_control_err3=self.last_control_err2
		self.last_control_err2=self.last_control_err
		self.last_control_err=self.cur_control_err
		self.cur_control_err=[self.target_pos[0]-x,self.target_pos[1]-y]
#         self.state=self.rl_last_act+self.cur_control_err+self.last_control_err+self.last_control_err2
		new_state=[]
		new_state.extend(self.rl_last_act)
		new_state.extend(self.cur_control_err)
		new_state.extend(self.last_control_err)
		new_state.extend(self.last_control_err2)
		new_state.extend(self.last_control_err3)
		new_state.extend(self.last_control_err4)
		self.state_buf2=new_state
		self.rl_last_act=act

		return np.reshape(np.array(self.state_buf2),[-1]), r, done


		# print("reward:",r)


	def test(self):
		self.set_target_pos((1,1,1))
		print("==============gaussian score==============")
		print(self.get_gaussian_score((1,1,1)))
		print(self.get_gaussian_score((1,0.5,1)))
		print(self.get_gaussian_score((0.5,0.5,1)))
		print("==============calculate reward==============")
		print(self.calculate_reward((1,1,1),(0,0,1)))
		print(self.calculate_reward((1,1,1),(1,1,1)))
		print(self.calculate_reward((1,2,1),(1,1,1)))
		print("==============reachable==============")
		print(self.reachable((1,1,1)))
		print(self.reachable((1,1.4,1)))
		print(self.reachable((1,2,1)))


	
if __name__ == '__main__':
    e=Env(10)
    e.test()

