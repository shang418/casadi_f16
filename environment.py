from casadi_env.aircraft.casadi_f16 import f16 
from RLcontroller_jsbsim.maneuvers.Trajectory import Trajectory
import casadi as ca

class CasadiEnv(): 
    """
    A class that wraps the casadi aircraft dynamics 
    for simulating an RL environment 
    
    """
    
    info_dict = {0: 'Good'
                 1: 'Crashed'
                }
                
    def __init__(self,traj: Trajectory, min_altitude: int = 0 , aircraft = f16, agent_interaction_freq: int = 10,DT_HZ = 100): 
    
        # set aircraft and integration time 
        self.aircraft = aircraft
        
        self.min_altitude = 0 
        # integration time and frequency of interaction 
        self.agent_interaction_HZ = agent_interaction_freq
        self.dt = 1/DT_HZ
        
        # get space objects 
        self.observation_space = len(self.aircraft.State().to_tuple())
        self.action_space = len(self.aircraft.Control().to_tuple())
        
    ################## FIXXXXXX THISSSSSSSS ####################
    def net_to_casadi(self,agent: NAF, ounoise): 
        
        nu_sym = ca.MX.sym('nu', self.action_space)
        x_sym = ca.MX.sym('x', self.observation_space)
        
        x = State.from_casadi(x_sym)
        nu = Control.from_casadi(nu_sym)
        
        return ca.Function('fcn_action',[x,nu],[agent.act(x,nu)])
        
    def step(self,t_start:int,state: nd.array, agent, ounoise): 
        
        """ runs one instance of the environment's dynamics given a function fcn_action 
        of the form f(t,x,mu) that predicts a control input u 
        
        """
        if not (state.shape[0] == self.observation_space):
            raise ValueError('mismatch between state and state space size')
        
        fcn_action = self.net_to_casadi(agent,ounoise)
        
        reward = self.getreward(state) # get reward based on current state
        
        next_state = self.sim_step(t_start,state,fcn_action) # simulate next state based on action 
        
        done, status = self.isdone(next_state) # check to see if simulation has crashed and info displays state of system 
        
        if status==1: # 1 - system crashed  
            next_state = None 
            
        return next_state, reward, done, self.info_dict.get(status)
    
    def getreward(self): 
        
        assert state is not None 
        
        r = self.traj.getreward(state)
        
        return r 
    
    def sim_simulate(self,t_start,state,fcn_action): 
    
        
        t_final = self.dt*self.agent_interaction_HZ + t_start
        
        data = self.aircraft.simulate(x0 = state, fcn_action, p = self.aircraft.Parameters(), \ 
                                       t0 = t_start, tf = t_final, dt = self.dt)
        
        # transform data to nd.array .. need to find type of data 
        
        #state_hist = data .. option to return state history for plotting 
        
        next_state = data[-self.observation_space:]
        
        return  next_state, 
        
    def isdone(self,next_state): 
        
        done = False 
        status = 0 
        
        if next_state.to_dict()['alt'] <= self.min_alt: 
            status = 1 
            done = True 
        
        #check to see if rates are within allowable bounds 
        
        return done, status
        
        
        