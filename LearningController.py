from environment import Env
# from AgentA2C import A2C_Online
from replay_buffer import ReplayBuffer
from visualization import *
from AgentDDPG import *
import numpy as np
import time
import copy
import rospy
import json
import tensorflow as tf
class LearningController:
	def __init__(self):
		self.env=None
		self.agent=None




	

	def start(self):

		batch_size = 128
		memory_warmup = batch_size*2
		max_explore_eps = 50

		t_max=2000
		gamma = 0.95
		tau = 0.001
		self.env=Env(height=3)
		n_observation=self.env.n_observation
		n_action=self.env.n_action
		sessionfilename="./DDPG_net_Class99.ckpt"
		tf.reset_default_graph()
		
		actorAsync = AsyncNets(n_observation,n_action,'Actor')
		actor,actor_target = actorAsync.get_subnets()
		criticAsync = AsyncNets(n_observation,n_action,'Critic')
		critic,critic_target = criticAsync.get_subnets()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		noise = UONoise(n_action)
		memory = Memory(memory_size= 100000)
		rewards=[]
		json.encoder.FLOAT_REPR= lambda o: format(o,'.4f')
		with tf.Session() as sess:
			try:
				saver.restore(sess, sessionfilename)
				# init.run()
			except Exception as e:
				print("!!!!!!!!!!!!!Load param failure!!!!!!!!!!!!!!!!!!!!1")
				print(e)
				init.run()
			actorAsync.set_session(sess)
			criticAsync.set_session(sess)

			for episode in range(500):
				total_reward = 0.0
				obs = self.env.reset()
				watcher=-1
				echo_dict={}
			
				# traj_x,traj_y,traj_x2,traj_y2=[],[],[],[]

				track_err=0
				self.env.ROSNode.log_axis_clear()
				self.env.ROSNode.log_xy_clear()

			

				for tstep in range(t_max):
					watcher=tstep
					x,y=self.env.target_pos[0],self.env.target_pos[1]

					uav_pos=self.env.ROSNode.get_uav_pos()[0:3]
					# traj_x2.append(x)
					# traj_y2.append(y)
					# traj_x.append(uav_pos[0])
					# traj_y.append(uav_pos[1])
					self.env.ROSNode.log_xy_add("command_pos",(x,y))
					self.env.ROSNode.log_xy_add("real_pos",(uav_pos[0],uav_pos[1]))

					

					track_err+=((x-uav_pos[0])**2+(y-uav_pos[1])**2)**0.5
					self.env.ROSNode.log_axis_add("error_x",(tstep,x-uav_pos[0]))
					self.env.ROSNode.log_axis_add("error_y",(tstep,y-uav_pos[1]))
					self.env.ROSNode.log_axis_add("error_abs",(tstep,((x-uav_pos[0])**2+(y-uav_pos[1])**2)**0.5))



					            
            		
					action = actor.predict_action(np.reshape(obs,[1,-1]))[0]
					
					p = episode/max_explore_eps
					p=np.clip(p,0.5,0.99)
					p=0.8

					action = action*p + (1-p)*next(noise)
					# action=np.array([0,0])


					self.env.ROSNode.log_axis_add("action_x",(tstep,action[0]))
					self.env.ROSNode.log_axis_add("action_y",(tstep,action[1]))
					self.env.ROSNode.log_axis_add("action_abs",(tstep,np.linalg.norm(action)))


					start_time=rospy.get_time()
					next_obs,reward,done=self.env.step(action)
					time_step=rospy.get_time()-start_time
					if done:
						break						
					# print(obs.shape,next_obs.shape)
					self.env.ROSNode.log_axis_add("reward",(tstep,reward))

					memory.append([obs,action,reward,next_obs,done])

					echo_dict["a"]=action.tolist()
					echo_dict["r"]=reward
					echo_dict["state"]=str(obs.tolist())
					echo_dict["next_state"]=str(next_obs.tolist())
					
					start_time=rospy.get_time()
					# memory_warmup=99999999999999999
					if len(memory) >= memory_warmup :
						memory_batch = memory.sample_batch(batch_size)
						extract_mem = lambda k : np.array([item[k] for item in memory_batch])
						obs_batch = extract_mem(0)
						action_batch = extract_mem(1)
						reward_batch = extract_mem(2)
						next_obs_batch = extract_mem(3)
			 			done_batch = extract_mem(4)		
						action_next = actor_target.predict_action(next_obs_batch)			 	

						Q_next = critic_target.predict_Q(next_obs_batch,action_next)[:,0]
						Qexpected_batch = reward_batch + gamma*(1-done_batch)*Q_next # target Q value
						Qexpected_batch = np.reshape(Qexpected_batch,[-1,1])
							
						critic.train(obs_batch,action_batch,Qexpected_batch)
			            # train actor
						action_grads = critic.compute_action_grads(obs_batch,action_batch)
			
						actor.train(obs_batch,action_grads)
			            # async update
						actorAsync.async_update(tau)
						criticAsync.async_update(tau)
						total_reward+=reward

					time_train=rospy.get_time()-start_time
					obs = next_obs

					echo_dict["info"]={"epi":episode,"tstep":tstep,"p":p,"time_step":time_step,"time_train":time_train}
					if tstep%5==0:
						js=json.dumps(echo_dict,sort_keys=True,indent=4,separators=(',',':'))

					rospy.loginfo_throttle(1,js)
					
		

				print("episode end at step: ",watcher)
				# filename="img/"+time.strftime("%d-%H:%M:%S",time.localtime())+".png"
				title=str("mean reward: ")+str(total_reward/watcher) + str("mean error: ")+str(track_err/watcher)
				self.env.ROSNode.log_showfigure(title)
				# plot_curve(traj_x,traj_y,traj_x2,traj_y2,title,filename)
				saver.save(sess,sessionfilename)
				saver.save(sess,"./meta_spec.ckpt")


	def test(self):
		x=[]
		y=[]
		for i in range(10000):
			a,b=self.curve(i/10.0)
			x.append(a)
			y.append(b)
		filename="img/"+time.strftime("%d-%H:%M:%S",time.localtime())+".png"
		title=str("mean reward: ")+str(2.0/3)
		plot_curve(x,y,x,y,title,filename)

if __name__ == '__main__':
	ctrl = LearningController()
	ctrl.start()
	#ctrl.test()