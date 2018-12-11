import os
import subprocess
import rospy
import time
from nav_msgs.msg import Odometry
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, State, WaypointList
from geometry_msgs.msg import PoseStamped, TwistStamped
from geometry_msgs.msg import Twist, Vector3Stamped, Pose,Point,Quaternion,Vector3
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Header
from utils import *
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# from geometry_msgs.msg import Twist, Vector3Stamped, Pose
# from mavros_msgs.srv import CommandBool, SetMode
# from environment import Env
# from QLearningAgent import QLearningAgent
"""

gazebo----ENV: -----------learning
			|s,r               |
		  statemachine-------s,r,a--


"""
class LogDict(dict):
	def __init__(self):
		pass


	def __setitem__(self,key,val):

		val_x=val[0]
		val_y=val[1]
		if key not in self:
			dict.__setitem__(self,key,{"x":[val_x],"y":[val_y]})
		else:
			dict.__getitem__(self,key)["x"].append(val_x)
			dict.__getitem__(self,key)["y"].append(val_y)

	def __getitem__(self,key):
		d=dict.__getitem__(self,key)
		return d["x"], d["y"]


class ROSNode:
	"""StateMachine: flight routinue control"""
	def __init__(self):

		self.rate=10
		self.rateobj = None
		self.set_arm=None
		self.set_mode=None
		self.L=1
		self.H=3	
		self.local_pos_sub = None
		
		self.set_arm= None
		self.set_mode = None
		self.pos_setpoint_pub = None
		self.target_odom_pub=None
		self.setpoint_odom_pub=None
		self.logxy_dict=LogDict()
		self.logaxis_dict=LogDict()


	def log_xy_add(self,key,val):
		self.logxy_dict[key]=val
	def log_xy_clear(self):
		self.logxy_dict.clear()
	def log_axis_add(self,key,val):
		self.logaxis_dict[key]=val
	def log_axis_clear(self):
		self.logaxis_dict.clear()
	
	def log_showfigure(self,title):
		dirname="img/"+time.strftime("%d-%H:%M:%S",time.localtime())
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		for keys in self.logaxis_dict.keys():
			x,y=self.logaxis_dict[keys]
			plot=plt.plot(x,y)
			fig, ax = plt.subplots() 
			ax.plot(x,y)
			plt.title(title)
			ax.autoscale()
			fig.savefig(os.path.join(dirname,keys))
			
		fig, ax = plt.subplots() 
		plt.title(title)
		for keys in self.logxy_dict.keys():
			x,y=self.logxy_dict[keys]
			ax.plot(x,y)
			
				
		ax.autoscale()
		fig.savefig(os.path.join(dirname,"xy_plots.png"))		


	def ros_delay(self,sec):
		rospy.sleep(sec)
		# for i in range(int(sec*self.rate+0.5)):
		# 	# rospy.loginfo("delaying" )
		# 	self.rateobj.sleep()
	def ros_interact_step(self, t):
		if t!=None:
			rospy.sleep(t)
		else:
			rospy.sleep(1.0/self.rate)

		
	def start_sim(self):
		self.err_cmd=0

		mavros = subprocess.Popen([r'roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557" > /dev/null '],shell=True)
		self.ros_delay(1)
		gazebo = subprocess.Popen([r"cd ~/src/Firmware/ && HEADLESS=1 make posix_sitl_default gazebo > /dev/null"],shell=True)
		# offb = subprocess.Popen([r'roslaunch offb offb_local.launch'],shell=True)
		# rviz = subprocess.Popen([r'rosrun rviz rviz'],shell=True)
		rospy.loginfo("start sim!")

	def stop_sim(self):
		subprocess.call([r'killall -9 mavros'],shell=True)
		subprocess.call([r'killall -9 px4'],shell=True)
		# subprocess.call([r'pkill rviz'],shell=True)
		subprocess.call([r'killall -9 gzserver'],shell=True)	
		subprocess.call([r'killall -9 gazebo'],shell=True)	

	def reset_sim(self):
		rospy.loginfo("============reset simulation===============")
		self.stop_sim()
		self.start_sim()

	def publish_target_point(self,x,y,z):
		odom=Odometry()

		odom.header.stamp = rospy.Time.now()
		odom.header.frame_id = "map"
    	
    	# set the position
		odom.pose.pose = Pose(Point(x, y, z), Quaternion(0,0,0,1))

    	# set the velocity
		odom.child_frame_id = "base_link"
		odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

		self.target_odom_pub.publish(odom)	

	def publish_setpoint_point(self,x,y,z):
		odom=Odometry()

		odom.header.stamp = rospy.Time.now()
		odom.header.frame_id = "map"
    	
    	# set the position
		odom.pose.pose = Pose(Point(x, y, z), Quaternion(0,0,0,1))

    	# set the velocity
		odom.child_frame_id = "base_link"
		odom.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))

		self.setpoint_odom_pub.publish(odom)	
	def set_velocity(self,vx,vy,vz):
		vel = TwistStamped()
		vel.twist.linear.x = vx
		vel.twist.linear.y = vy
		vel.twist.linear.z = vz
		vel.header = Header()
		vel.header.frame_id = "map"
		vel.header.stamp = rospy.Time.now()
		self.pos_setvel_pub.publish(vel)

	def moveto(self,x,y,z):
		
		self.publish_setpoint_point(x,y,z)

		pos = PoseStamped()
		pos.pose.position.x = x
		pos.pose.position.y = y
		pos.pose.position.z = z
		pos.header = Header()
		pos.header.frame_id = "map"
		pos.header.stamp = rospy.Time.now()
		# rospy.loginfo_throttle(0.41,pos)
		# rospy.loginfo(pos)
		self.pos_setpoint_pub.publish(pos)
		# self.publish_target_odom.publish(x,y,z)

	def local_position_callback(self,data):
		self.local_position = data	
		# rospy.loginfo_throttle(1,data)

	def get_uav_pos(self):
		rx=self.local_position.pose.position.x
		ry=self.local_position.pose.position.y
		rz=self.local_position.pose.position.z
		qx=self.local_position.pose.orientation.x
		qy=self.local_position.pose.orientation.y
		qz=self.local_position.pose.orientation.z
		qw=self.local_position.pose.orientation.w

		return rx,ry,rz,qx,qy,qz,qw

	def hover(self, sec):
		for i in range(int(sec/self.rate)):
			self.moveto(0,0,self.H)
			self.ros_interact_step()

	def flight_init(self):
		#https://github.com/ethz-asl/rotors_simulator/blob/master/rqt_rotors/src/rqt_rotors/hil_plugin.py
		#https://github.com/PX4/Firmware/blob/master/integrationtests/python_src/px4_it/mavros/mavros_offboard_posctl_test.py
		#https://github.com/PX4/Firmware/blob/master/integrationtests/python_src/px4_it/mavros/mavros_test_common.py
		rospy.loginfo("============flight_init===============")
		for i in range(5):

			try:
		
				for i in range(10):
					self.moveto(0,0,self.H)
	
				self.ros_delay(0.1)
				res = self.set_arm(True)
				if not res.success:
					raise rospy.ServiceException("failed to send ARM command")
					continue
				res = self.set_mode(0,"OFFBOARD")
				if not res.mode_sent:
					raise rospy.ServiceException("failed to send OFFBOARD command")
					continue

			except rospy.ServiceException as e:
				rospy.logerr(e)
				continue
			return True

		
			
		return False

	def test_position_control(self,tarpos,maxspd=2):
		Kp=1
		vx=Kp*(tarpos[0]-self.get_uav_pos()[0])
		vy=Kp*(tarpos[1]-self.get_uav_pos()[1])
		vz=Kp*(tarpos[2]-self.get_uav_pos()[2])

		norm=np.linalg.norm([vx,vy,vz])
		if norm>maxspd:
			vx=vx*maxspd/norm
			vy=vy*maxspd/norm
			vz=vz*maxspd/norm

		self.set_velocity(vx,vy,vz)


	def start(self):
		print("calling start")
		#references: 
		rospy.init_node('ROSNode')
		rospy.loginfo("ROSNode started!")
		self.rateobj=rospy.Rate(self.rate)
		self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose', 
												PoseStamped,
												self.local_position_callback)

		self.set_arm=rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
		self.set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
		self.pos_setpoint_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
		self.pos_setvel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel',TwistStamped,queue_size=10)
		self.target_odom_pub = rospy.Publisher("target_odom", Odometry, queue_size=50)
		self.setpoint_odom_pub = rospy.Publisher("setpoint_odom", Odometry, queue_size=50)
		self.local_position=None


		state = 0

		while not rospy.is_shutdown():

			if state==0:
				self.reset_sim()
				self.ros_delay(10)#delay should be long enough... or potential bug..
				res = self.flight_init()
				if res:
					rospy.loginfo("flight init ok!")
					state = 1
				else:
					state=0 #do nothing
					# self.reset_sim()
					# self.ros_delay(5)
			if state==1:
				# self.Env.moveto(0,0,self.H)

				cnt=0
				print("taking off....")
				while True:
					try:
						rx,ry,rz,_,_,_,_=self.get_uav_pos()
						# self.set_velocity(0,0,10)
						self.moveto(0,0,self.H)
					except:
						cnt+=1
						print("error 0999!!!!!")
						continue
						
					if get_dis(rx,ry,rz,0,0,self.H)<0.3:
						state=99
						break
					else:
						cnt+=1
					# print("taking off....",[rx,ry,rz])
					
					self.ros_delay(0.05)
					rospy.loginfo_throttle(1,(rx,ry,rz))

					if cnt>=300:
						state=0
						break

			if state==99:
				rospy.loginfo("============take off ok!============")
				return True
	

			self.rateobj.sleep()

 	def start_test(self):
		print("calling start")
		#references: 
		rospy.init_node('ROSNode')
		rospy.loginfo("ROSNode started!")
		self.rateobj=rospy.Rate(self.rate)
		self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose', 
												PoseStamped,
												self.local_position_callback)

		self.set_arm=rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
		self.set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
		self.pos_setpoint_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
		self.pos_setvel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel',TwistStamped,queue_size=10)
		self.target_odom_pub = rospy.Publisher("target_odom", Odometry, queue_size=50)
		self.setpoint_odom_pub = rospy.Publisher("setpoint_odom", Odometry, queue_size=50)
		self.local_position=None


		state = 0

		while not rospy.is_shutdown():

			if state==0:
				self.reset_sim()
				self.ros_delay(10)#delay should be long enough... or potential bug..
				res = self.flight_init()
				if res:
					rospy.loginfo("flight init ok!")
					state = 1
				else:
					state=0 #do nothing
					# self.reset_sim()
					# self.ros_delay(5)
			if state==1:
				# self.Env.moveto(0,0,self.H)

				cnt=0
				print("taking off....")
				while True:
					try:
						rx,ry,rz,_,_,_,_=self.get_uav_pos()
						# self.set_velocity(0,0,10)
						self.moveto(0,0,self.H)
					except:
						cnt+=1
						print("error 0999!!!!!")
						continue
					
					if get_dis(rx,ry,rz,0,0,self.H)<0.3:
						state=2
						
						break
					else:
						cnt+=1
					# print("taking off....",[rx,ry,rz])
					
					self.ros_delay(0.05)
					rospy.loginfo_throttle(1,(rx,ry,rz))

					if cnt>=300:
						state=0
						break
			if state==2:

				self.test_position_control((-1,-2,5))
				self.ros_delay(0.05)
				rx,ry,rz,_,_,_,_=self.get_uav_pos()
				rospy.loginfo_throttle(1,(rx,ry,rz))


			if state==99:
				rospy.loginfo("============take off ok!============")
				return True
	

			self.rateobj.sleep()

	def test(self):
		self.reset_sim()
		rospy.init_node('ROSNode')
		rospy.loginfo("ROSNode started!")
		self.rateobj=rospy.Rate(1)
		while True:

			seconds = rospy.get_time()
			clk=time.clock()
			self.rateobj.sleep()
			x=rospy.get_time()-seconds
			x2=time.clock()-clk
			print("**************",x,x2,"**************")
		self.stop_sim()












if __name__ == '__main__':
	s=ROSNode()
	s.start_test()
	
