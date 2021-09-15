import numpy as np
from gym import utils, spaces
import gym
import torch
import pydiffarti as pd
import api_diff
import os 

def calc_jac(y, x, free=False):
    ans = []
    py = y.reshape([-1])
    for i in range(py.shape[0]):
        flag = True
        if i == py.shape[0]-1 and free:
            flag = False
        ans.append(torch.autograd.grad(py[i], x, retain_graph=flag)[0])
    ans = torch.stack(ans, dim=0)
    return ans.reshape(y.shape+x.shape)

class PendulumOursEnv(gym.Env, utils.EzPickle):
    def __init__(self, n_links=5, use_term=0, need_jac=True):
        utils.EzPickle.__init__(self)
        world = pd.TinyWorld(do_vis=False)
        parser = pd.TinyUrdfParser()
        convert_tool = pd.UrdfToMultiBody2()

        mb = world.create_multi_body()
        mb.isFloating = False
        self.diffarti_path = '../../../../../..'
        urdf_structures = parser.load_urdf('{}/data/pendulum{}.urdf'.format(self.diffarti_path, n_links))
        convert_tool.convert2(urdf_structures, world, mb)

        init_q = torch.zeros([n_links], dtype=torch.float32)
        q_lo = -torch.ones([n_links], dtype=torch.float32) * np.pi * 2
        q_hi = torch.ones([n_links], dtype=torch.float32) * np.pi * 2
        init_qd = torch.zeros([n_links], dtype=torch.float32)

        grav = pd.TinyVector3(pd.Utils.zero(), pd.Utils.zero(), pd.Utils.fraction(-98, 10))

        dt = 1./60
        finer = 1
        world.dt = pd.Utils.scalar_from_double(dt/finer)

        self.output_file = None
        self.world = world
        self.need_jac = need_jac
        self.dt = dt
        self.finer = finer
        self.wdt = self.dt / self.finer
        self.obs_siz = 1+n_links+3*(n_links+1)
        self.act_siz = n_links
        self.init_q = init_q
        self.init_qd = init_qd
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.clip = 10.
        self.power = 10.
        self.limit = 10.
        self.target_height = 0.3
        self.electricity_cost = -0.2
        self.stall_torque_cost = -0.1
        self.joints_at_limit_cost = -0.1
        self.init_z = 1
        self.mb = mb
        self.grav = grav
        world.adj_initialize(grav, 1000, self.act_siz)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=[self.obs_siz])
        self.action_space = spaces.Box(low=-1, high=1, shape=[self.act_siz])
        self.target_joint = torch.tensor([0.,0.,n_links*0.5], dtype=torch.float32)
        self.n_links = n_links
        self.use_term = use_term

    def get_ext(self, q0, qd, tau):
        return tau
        q = q0 + qd * self.wdt
        dist0 = torch.min(self.q_hi - q, torch.zeros([self.n_links]))
        dist1 = torch.max(self.q_lo - q, torch.zeros([self.n_links]))
        force = torch.min(torch.max(10 * (dist0 + dist1), torch.tensor(-5)), torch.tensor(5))
        force = force * self.power
        return tau + force


    def RK4(self, q, qd, tau, world):
        t1 = self.get_ext(q, qd, tau)
        r10, r11 = api_diff.sim_layer(q, qd, t1, world)

        q10=((q+r10)/2); q11=((qd+r11)/2)
        t2 = self.get_ext(q10, q11, tau)
        r20, r21 = api_diff.sim_layer(q10, q11, t2, world)

        q20=((q+r20)/2); q21=((qd+r21)/2)
        t3 = self.get_ext(q20, q21, tau)
        r30, r31 = api_diff.sim_layer(q20, q21, t3, world)

        q30=r30; q31 = r31
        t4 = self.get_ext(q30, q31, tau)
        r40, r41 = api_diff.sim_layer(q30, q31, t4, world)

        return (r20+r30+r40+3*q)/6, (r21+r31+r41+3*qd)/6

    def is_done(self):
        return self.frame >= 100 or torch.isnan(self.reward) or not self.success

    def get_state(self):
        joints = api_diff.get_joints(self.q, self.world)
        self.state = torch.cat([joints[3:], torch.tensor([self.frame/100.])], dim=0)
        self.state = torch.cat([self.qd, self.state], dim=0)
        return self.state

    def get_reward(self):
        cur_dist = (api_diff.get_joints(self.q, self.world)[-3:] - self.target_joint).norm()**2
        if self.is_done():
            progress = (1-self.use_term) * (self.prev_dist - cur_dist) - cur_dist
        else:
            progress = (1-self.use_term) * (self.prev_dist - cur_dist) #/ self.total
        self.prev_dist = cur_dist
        self.reward = progress
        if self.output_file is not None:
            print('r: ',self.reward, cur_dist)
        return self.reward

    def step(self, action):
        # print('[ ENV ]: action: ',action)
        if self.output_file is not None:
            for i in range(len(self.vert)):
                self.print_link(i)
        self.frame += 1
        if False:
            action = action * 0 - 1
            print(self.q)
            if self.frame == 99:
                while True:
                    continue
        self.action = action
        self.success = False
        tau = torch.tensor(action, dtype=torch.float32) #torch.clamp(action[0],-1,1) * self.power
        tau.requires_grad = True
        self.q = self.q.detach()
        self.qd = self.qd.detach()
        for substep in range(self.finer):
            self.q, self.qd = api_diff.sim_layer(self.q, self.qd, tau * self.power, self.world)
            # self.q, self.qd = self.RK4(self.q, self.qd, tau * self.power, self.world)
        self.success = True
        self.tau = tau
        self.get_state()
        self.get_reward()
        if torch.isnan(self.q).any():
            print('nan!')
            import sys;sys.exit(0)
        if self.need_jac:
            jac = calc_jac(self.state, self.tau).numpy()
            jacr = calc_jac(self.reward, self.tau, free=True).reshape([1,-1]).numpy()
        else:
            ind = 0
            # jac = np.zeros([1, self.act_siz])
            jac = np.zeros([self.obs_siz, self.act_siz])
            jacr = np.zeros([1,self.act_siz])
        return self.state.detach().numpy(), self.reward.detach().numpy(), self.is_done(), {
            'jacfull': jac, 'jacr': jacr, 'jacind': np.array([-1])}

    def addvert(self, i):
        ans = []
        face = []
        self.fcs.append(face)
        self.vert.append(ans)

    def print_link(self, i):
        with open(self.output_file + '/{}_{}.obj'.format(self.frame, i), 'w') as f:
            for vert in self.vert[i]:
                v = pd.TinyVector3(pd.Utils.scalar_from_double(vert[0]),
                        pd.Utils.scalar_from_double(vert[1]),
                        pd.Utils.scalar_from_double(vert[2]))
                world = self.mb.body_to_world(i, v)
                out = [pd.Utils.getDouble(world[0]),pd.Utils.getDouble(world[1]),pd.Utils.getDouble(world[2])]
                f.write('v {} {} {}\n'.format(out[0],out[1],out[2]))
            for face in self.fcs[i]:
                f.write('f {} {} {}\n'.format(face[0],face[1],face[2]))

    def reset(self):
        if self.output_file is not None:
            print('GOOD!',len(self.mb.links))
            self.vert = []
            self.fcs = []
            for i in range(len(self.mb.links)-1):
                self.addvert(i)
        self.q = self.init_q + 0
        self.qd = self.init_qd + 0
        self.frame = 0
        self.reward = torch.tensor(0.)
        self.success = True
        self.prev_dist = (api_diff.get_joints(self.q, self.world)[-3:] - self.target_joint).norm()**2
        self.total = self.prev_dist
        return self.get_state().numpy()

