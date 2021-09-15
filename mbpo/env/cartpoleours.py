import numpy as np
from gym import utils, spaces
import gym
import torch
import pytinydiffsim as pd
import api_diff

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

class CartPoleOursEnv(gym.Env, utils.EzPickle):
    """
        External forces (sim.data.cfrc_ext) are removed from the observation.
        Otherwise identical to Ant-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py
    """
    def __init__(self):
        # print(n_links)
        # while True:
        #     continue
        # mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)
        world = pd.TinyWorld(do_vis=False)
        parser = pd.TinyUrdfParser()
        convert_tool = pd.UrdfToMultiBody2()

        mb = world.create_multi_body()
        mb.isFloating = False
        urdf_structures = parser.load_urdf('/scratch1/diffarti/data/cartpole.urdf')
        convert_tool.convert2(urdf_structures, world, mb)

        n_links = 2

        init_q = torch.zeros([n_links], dtype=torch.float32)
        init_qd = torch.zeros([n_links], dtype=torch.float32)

        grav = pd.TinyVector3(pd.Utils.zero(), pd.Utils.zero(), pd.Utils.fraction(-98, 10))

        dt = 1./60
        finer = 1
        world.dt = pd.Utils.scalar_from_double(dt/finer)

        self.world = world
        self.dt = dt
        self.finer = finer
        self.obs_siz = n_links*2
        self.act_siz = 1
        self.act_siz_world = 2
        self.init_q = init_q
        self.init_qd = init_qd
        self.power = 100.
        self.limit = 300.
        self.target_height = 0.3
        self.electricity_cost = -0.2
        self.stall_torque_cost = -0.1
        self.joints_at_limit_cost = -0.1
        self.init_z = 1
        self.mb = mb
        self.grav = grav
        world.adj_initialize(grav, 1000, self.act_siz_world)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=[self.obs_siz])
        self.action_space = spaces.Box(low=-3, high=3, shape=[self.act_siz])
        # self.std = np.load('/scratch1/diffarti/tau.npy') / self.power
        # self.target_joint = torch.tensor([0.,0.,n_links*0.5], dtype=torch.float32)

    def is_done(self):
        return self.q[1].abs()>0.2 or torch.isnan(self.reward) or not self.success

    def get_state(self):
        self.state = torch.cat([self.q, self.qd], dim=0)
        # print('[ ENV ]: state shape=',self.state.shape)
        return self.state

    def get_reward(self):
        alive = self.q[0]*0+1
        if not self.success:
            alive = alive * 0 - 100
        self.reward = alive
        self.rewards = [alive]
        # print(api_diff.get_joints(self.q, self.world).reshape([-1,3]))
        return self.reward

    def step(self, action):
        # print('[ ENV ]: action: ',action)
        self.frame += 1
        self.action = action
        self.success = False
        # action = self.std[self.frame-1]
        tau = torch.tensor(action * self.power, dtype=torch.float32) #torch.clamp(action[0],-1,1) * self.power
        # print('?????? ',tau)
        tau = torch.clamp(tau, -self.limit, self.limit)
        tau.requires_grad = True
        # print('tau=',tau)
        try:
            for substep in range(self.finer):
                self.q, self.qd = api_diff.sim_layer(self.q, self.qd, torch.cat([tau, torch.zeros([1])], axis=0), self.world)
            self.success = True
        except:
            pass
        # print(self.q, tau)
        self.q = torch.stack([torch.clamp(self.q[0], -1, 1), self.q[1]], dim=0)
        # print('q=',self.q,self.qd)
        # self.q = self.q.detach()
        # self.qd = self.qd.detach()
        self.tau = tau
        # print('[ ENV ]: action done')
        self.get_state()
        self.get_reward()
        jac = calc_jac(self.state, self.tau).numpy()
        jacr = calc_jac(self.reward, self.tau, free=True).reshape([1,-1]).numpy()
        # print('??????',self.std.shape,jac.shape,self.tau.shape)
        # print(jac.shape,jacr.shape)
        # import sys;sys.exit(0)
        return self.state.detach().numpy(), self.reward.detach().numpy(), self.is_done(), {
            'jac': jac, 'jacr': jacr}

    def reset(self):
        self.q = self.init_q + torch.rand(2)*0.02-0.01
        self.qd = self.init_qd + torch.rand(2)*0.02-0.01
        self.frame = 0
        self.reward = torch.tensor(0.)
        self.success = True
        return self.get_state().numpy()

