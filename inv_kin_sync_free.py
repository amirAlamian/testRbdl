import sys
import os
userdir = os.path.expanduser('~')
sys.path.append(userdir+"/projects/actuator")
from actuator import Actuator
from homing import home

sys.path.append(userdir+"/projects/rbdl3/build/python")
import rbdl
from robot_class import ROBOT
import time
import numpy as np
from numpy import sin, cos, pi, zeros, array, sqrt
from numpy.linalg import norm, inv, cond
from numpy.linalg import matrix_rank as rank
import matplotlib.pyplot as plt
plt.close('all')


def fixq3(q):
        r = 18./28.
        q[-1] = q[-1]*r
        return q
    
def fixq3inv(q):
        r = 28./18.
        q[-1] = q[-1]*r
        return q
    
def robot2rbdl(p1, p2, p3):
    q = [-(p1 - xh1), -(p2 - xh2), -(p3 - xh3)]
    q = fixq3(q)
    return q

def rbdl2robot(p1, p2, p3):
    p1, p2, p3 = fixq3inv([p1, p2, p3])
    rx = [-p1 + xh1, -p2 + xh2, -p3 + xh3]
    return rx

def modelKin(q):
    J = robot.calcJc(np.array(q))
    p = robot.pose_end(np.array(q))
    return [p, J]



n   = 3 # number of links
path =  "leg_RBDL.urdf"
robot = ROBOT(np.zeros((3, 1)), np.zeros((3, 1)), path) #model

leg = Actuator('can0') #real robot
m1 = 0x01
m2 = 0x02
m3 = 0x05

kp = 40
kd = 1

leg.enable(m1)
leg.enable(m2)
leg.enable(m3)

x_home = home([leg, m1, m2, m3], kp=kp, kd=kd, enable_motors=False, disable_motors=False)

# kp = 60
# kd = 2
#kp=40
#kd=1

kp=40
kd=1

# kp=20
# kd=0.5


xh1 = x_home[0]
xh2 = x_home[1]
# xh3 = x_home[2] - 3.88 # different zero position
# xh3 = x_home[2] - 1.16 # different zero position
xh3 = x_home[2] - 3.7132


rx1 = leg.command(m1, xh1, 0, kp, 0, 0)
rx2 = leg.command(m2, xh2, 0, kd, 0, 0)
rx3 = leg.command(m3, 0, 0, 0, 0, 0) #TODO
#f=(rx1[1],rx2[1],rx3[1])
#print ("rx", f)
q = robot2rbdl(rx1[1], rx2[1], rx3[1])
qr_pre = q
q_home = q
#print ("q_rbdl_home", q)
x = robot.pose_end(np.array(q))
home_pos  = x
print ("Homing position", home_pos)


Posi,Ji = modelKin(q)

dt = 0.008
tf  = 40;
k   = 3   # task space dimension

tvec  = np.arange(0, tf, dt)
w = np.pi/2
A = .1
xxd = home_pos[0] + A*np.sin(w*tvec ) 
dxxd = A*w*np.cos(w*tvec)

tpre = time.time()

# ref Jacobian
#TODO
Qr  = q_home
Pos, J = modelKin(Qr)
J      = J[0:k,:] 
Jr = np.zeros(k)

for i in range(k):
    Jr[i]=norm(J[i,:])**2

Jr_diag = np.diag(Jr)


# IK Gain
K_IK= 1.         # IK close loop gain
D_IK= .001     # IK damping gain
MaxI= 10         # Maximum number of iteration

# Position limits
Qmax = np.array([q_home[0] + pi/2, q_home[1] + pi/4, q_home[2] + pi/4])
Qmin = np.array([q_home[0] - pi/2, q_home[1] - pi/4, q_home[2] - pi/4])

# Velocity limits
Qdmax = 4*np.array([1., 1., 1.])
v_eps = 1e-3
Qdmin = -Qdmax;

# Acceleration limits
Qddmax = 10 * np.array([1., 1., 1.])
Qddmin = -Qddmax


# IK variables definitions
Smin    = zeros((1,n))
Smax    = zeros((1,n))

# Saving Variables
dim = int(tf/dt)
TrajV  = zeros((dim,k))
PosV   = zeros((dim,k))
QV      = zeros((dim,n))
QdV     = zeros((dim,n))
CV      = zeros((dim,1))
sV      = zeros((dim,1))
tV      = zeros((dim,1))
jV      = zeros((dim,1))
CJNV    = zeros((dim,MaxI))   # critical joint number
motor1_res = zeros((dim,1))
motor2_res = zeros((dim,1))
motor3_res = zeros((dim,1))
WM = np.diag(np.ones(n))




def IK(Q, Traj, Dpd):
    
    # Jacobian/Position extraction
    Pos,J = modelKin(Q);
    J      = J[:k,:]        # only position related block    
    JJ      = J
    Pos	= Pos[:k]
        
#     velocity vector
    Pos_error = Traj - Pos
    Xd = Dpd + K_IK * Pos_error
    
    
    # update the joint limits
    Vmin   = np.max(np.vstack(((Qmin-Q)/dt,Qdmin,-sqrt(Qddmin * (Qmin-Q)))), axis=0)    
    Vmax   = np.min(np.vstack(((Qmax-Q)/dt,Qdmax, sqrt(Qddmax * (Qmax-Q)))), axis=0)    
    
#     IK
#     damping
    ED = 1 / 2 * (Pos_error.dot(Pos_error))
    # initialisation
    SMv = np.ones((1,n))
    QdN = np.zeros((1,n))
    s = 1
    # first solution
    Jtw = np.dot(JJ , np.diag(SMv[0]))
#%     Ji  = Jtw'/(ED * eye(2*k) + D_IK*diag([J1r,J2r]) + Jtw * Jtw'); 
    Jtw_T = np.transpose(Jtw)
    aa = np.dot(WM, Jtw_T) 
    bb = ED * np.eye(k) + D_IK * Jr_diag + np.dot(Jtw, aa)  

    Ji = np.dot(aa, inv(bb))    

    for j in range(MaxI):
        Qdg = s* np.dot(Ji, np.transpose(Xd))
        Qdn = (QdN.T + np.dot(Ji , np.dot( - JJ , QdN.T))).flatten()
        Qd  = Qdg + Qdn
        for ii in range(n):
            if Qd[ii] > Vmax[ii] + v_eps:
                Smax[0,ii] = (Vmax[ii] - Qdn[ii])/Qdg[ii]
            elif Qd[ii] < Vmin[ii] - v_eps:
                Smax[0,ii] = (Vmin[ii] - Qdn[ii])/Qdg[ii];
            else:
                Smax[0,ii] =  np.inf; # it can be any number bigger than 1, e.g. 2
             

        CJN = np.argmin(Smax[0])
        sout = Smax[0, CJN]
        
#%         when there should be a saturated joint
        if sout<1:
#%             Saturating the velocity of Critiical Joint
            SMv[0,CJN] = 0
            if Qd[CJN] > Vmax[CJN]:
                QdN[0, CJN] = Vmax[CJN]
            else:
                QdN[0, CJN] = Vmin[CJN]

       
        if sout>1:  #% there is no limit exceedance 
#%             print('redundancy successfully limited the velocity')
            break
#            pass
#            %%%%%%% to be updated based on: initial rank of J and W update !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        elif rank(np.dot(JJ,np.diag(SMv[0]))) < k:
            SMv[0,CJN] = 1
            QdN[0,CJN] = 0
            s = sout
            #%             print('redundancy cannot limit the velocity, so the velocity is scaled')
        else:
#            % Jacobian update

#            %% Jacobian update
            Jtw = np.dot(JJ , np.diag(SMv[0]))
#            %     Ji  = Jtw'/(ED * eye(2*k) + D_IK*diag([J1r,J2r]) + Jtw * Jtw');         
            Jtw_T = np.transpose(Jtw)
            aa = np.dot(WM, Jtw_T) 
            bb = ED * np.eye(k) + D_IK * Jr_diag + np.dot(Jtw, aa)  

            Ji = np.dot(aa, inv(bb))                  
            
        jV[i,0] = j + 1
        CJNV[i,j] = CJN;
    
    return [Qd, s, Ji, Pos]


for i, t in enumerate(tvec):
    
    xd = np.array([xxd[i], home_pos[1], home_pos[2]])
   # xd = np.array([xxd[i], .0862, -.32])
    dxd = np.array([dxxd[i], 0, 0])
    
    dqr, s, Ji, Pos = IK(q, xd, dxd)  
    
    qr = dqr*dt + qr_pre
    
    diff = np.abs(qr - q)

    if (diff > .1).any() or np.linalg.cond(J) > 20:
        print("diff:")
        print(diff)
#         print (np.linalg.cond(J))
        rx1 = leg.command(m1, 0, 0, 0, 0, 0)
        rx2 = leg.command(m2, 0, 0, 0, 0, 0)
        rx3 = leg.command(m3, 0, 0, 0, 0, 0)
        leg.disable(m1)
        leg.disable(m2)
        leg.disable(m3)
        print ("Unsafe command to motors is detected!")
        break
        raise ValueError("Unsafe command to motors is detected!")
    
    qr_pre = qr
    
    rx = rbdl2robot(qr[0], qr[1], qr[2])
#     print ("rx: ", rx)
        
    rx1 = leg.command(m1, rx[0], 0, kp, kd, 0)
    rx2 = leg.command(m2, rx[1], 0, kp, kd, 0)
    rx3 = leg.command(m3, rx[2], 0, kp, kd, 0)
    
    
    q = robot2rbdl(rx1[1], rx2[1], rx3[1])
    x = robot.pose_end(np.array(q))
    
    #    %% Data saving
    motor1_res[i,:] = rx1[3]
    motor2_res[i,:] = rx2[3]
    motor3_res[i,:] = rx3[3]
    CV[i,:] = cond(Ji)
    TrajV[i,:] = xd
    PosV[i,:] = x
    QV[i,:] = q
    QdV[i,:] = dqr
#     tV[i,:] = t
    sV[i,:] = s
    
    tnow = time.time()
    if tnow - tpre > dt:
        print ("Warning! actual step time is bigger than dt!")
        print ("It is: ", tnow - tpre, " at time = ", tnow)
    while(time.time() - tpre < dt): temp = 0
    tpre = time.time()
    
    
    
    
    
    
    
    



# q = robot2rbdl(rx1[1], rx2[1], rx3[1])
# pose_tip = robot.pose_end(np.array(q))
# J = robot.calcJc(np.array(q))
    
    
if True:
    rx1 = leg.command(m1, 0, 0, 0, 0, 0)
    rx2 = leg.command(m2, 0, 0, 0, 0, 0)
    rx3 = leg.command(m3, 0, 0, 0, 0, 0)
    leg.disable(m1)
    leg.disable(m2)
    leg.disable(m3)

tV = tvec
#
plt.figure();
plt.subplot(211)
plt.plot(tV,TrajV,'-',tV,PosV,'--')
plt.title('Desired and actual trajectories')
plt.subplot(212)
plt.plot(tV,np.abs(TrajV - PosV))
plt.ylabel('error')
##
plt.figure()
plt.plot(tV,QdV)
plt.title('Reference Joint Velocities')
##
plt.figure()
plt.plot(tV,QV,'-')
plt.title('joint position')
#plt.legend('q1','q2','q3','q4','q5','q6','q7')
##
# plt.figure()
# plt.plot(tV,CJNV)
# plt.title('Critical Joint Number')
##
plt.figure()
plt.plot(tV,sV)
plt.title('Scaling factor')
##
# plt.figure()
# plt.plot(tV,jV)
# plt.title('Number of iterations')
##
plt.figure()
plt.plot(tV,CV)
plt.title('Jacobian Condition Number')
##
plt.figure();
plt.subplot(311)
plt.plot(tV,motor1_res)
plt.title('motor1 response')
plt.subplot(312)
plt.plot(tV,motor2_res)
plt.ylabel('motor2 response')
plt.subplot(313)
plt.plot(tV,motor3_res)
plt.ylabel('motor3 response')
##
plt.show()
