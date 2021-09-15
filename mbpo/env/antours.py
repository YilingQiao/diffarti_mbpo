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

class AntOursEnv(gym.Env, utils.EzPickle):
    """
        External forces (sim.data.cfrc_ext) are removed from the observation.
        Otherwise identical to Ant-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py
    """
    def __init__(self, enable_qlimit=True, need_jac=True, mode=2, clip=10, random_init=1):
        utils.EzPickle.__init__(self)
        world = pd.TinyWorld(do_vis=False)
        parser = pd.TinyUrdfParser()
        convert_tool = pd.UrdfToMultiBody2()

        plane_mb = world.create_multi_body()
        plane_mb.isFloating = False
        self.diffarti_path = '../../../../../..'
        plane_urdf_structures = parser.load_urdf('{}/data/plane_implicit.urdf'.format(self.diffarti_path))
        convert_tool.convert2(plane_urdf_structures, world, plane_mb)

        mb = world.create_multi_body()
        mb.isFloating = True
        urdf_structures = parser.load_urdf('{}/data/ant_torso1.urdf'.format(self.diffarti_path))
        convert_tool.convert2(urdf_structures, world, mb)

        deg90 = np.pi / 2
        init_q = torch.tensor([0.0,0.0,0.0,1.0, 0.0,0.0,0.75, 0,0,0,0,0,-0,0,-0], dtype=torch.float32)
        q_lo = torch.tensor([-30,30,-30,30,-30,-70,-30,-70], dtype=torch.float32) / 180. * np.pi
        q_hi = torch.tensor([30,70,30,70,30,-30,30,-30], dtype=torch.float32) / 180. * np.pi
        init_qd = torch.zeros([8+6], dtype=torch.float32)

        grav = pd.TinyVector3(pd.Utils.zero(), pd.Utils.zero(), pd.Utils.scalar_from_double(-9.81))

        dt = 1./100*5
        finer = 5
        world.dt = pd.Utils.scalar_from_double(dt/finer)

        self.output_file = None
        self.world = world
        self.dt = dt
        self.mode = mode
        self.clip = clip
        self.finer = finer
        self.wdt = self.dt / self.finer
        self.obs_siz = 8+5+8+6 #+8*3 #8+7+8+6 #
        self.act_siz = 8
        self.frame = 0
        self.init_q = init_q
        self.init_qd = init_qd
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.mult = 1
        self.power = 150. * self.mult
        self.limit = 150. * self.mult
        self.target_height = 0.3
        self.electricity_cost = -0.2
        self.stall_torque_cost = -0.1
        self.joints_at_limit_cost = -0.1
        self.mb = mb
        self.plane_mb = plane_mb
        self.grav = grav
        self.enable_qlimit = enable_qlimit
        self.need_jac = need_jac
        self.q = self.init_q + 0
        self.qd = self.init_qd + 0
        self.act_lr = 1
        self.random_init = random_init
        world.adj_initialize(grav, 1000, self.act_siz)
        world.friction = pd.Utils.scalar_from_double(1.)
        for l in mb.links:
            l.damping = pd.Utils.scalar_from_double(1.)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=[self.obs_siz])
        self.action_space = spaces.Box(low=-1, high=1, shape=[self.act_siz])

    def get_ext(self, q0, qd, tau):
        q = q0[-8:] + qd[-8:] * self.wdt
        if self.enable_qlimit:
            dist0 = torch.min(self.q_hi - q, torch.zeros([8]))
            dist1 = torch.max(self.q_lo - q, torch.zeros([8]))
            force = torch.min(torch.max(10 * (dist0 + dist1), torch.tensor(-5)), torch.tensor(5))
            force = force * self.power / self.mult
            return tau + force 
        return tau


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
        # return True
        return (self.q.detach().numpy())[6] < 0.2 or (self.q.detach().numpy())[6] > 1.0 or torch.isnan(self.q).any() or not self.success

    def get_state(self):
        self.state = torch.cat([
            self.q[:4],
            self.q[6:],
            self.qd*0.05,
        ], dim=0)
        absstate = self.state.abs().detach().numpy()
        return self.state, [absstate[:4],absstate[4:13],absstate[13:27]]#,absstate[27:]]

    def get_reward(self, a):
        xposafter = api_diff.get_com(self.q, self.world)[3] #self.q[4]
        forward_reward = (xposafter - self.xposbefore)/self.dt
        self.xposbefore = xposafter.detach()
        ctrl_cost = .5 * (a**2).sum()
        contact_cost = 0 #0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        self.reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        if torch.isnan(self.reward):
            self.reward *= 0
            self.success = False
        if self.output_file is not None:
            print('r: ',self.reward, self.q[4])
        return self.reward, [forward_reward.detach().numpy(), ctrl_cost.detach().numpy()]

    def step(self, action):
        if self.output_file is not None:
            for i in range(len(self.vert)):
                self.print_link(i)
        self.frame += 1
        self.action = action
        self.success = False

        tau = torch.tensor(action, dtype=torch.float32)
        tau.requires_grad = True
        self.q = self.q.detach()
        self.qd = self.qd.detach()
        for substep in range(self.finer):
            self.q, self.qd = self.RK4(self.q, self.qd, tau * self.power, self.world)
        self.success = True
        self.tau = tau
        # print('[ ENV ]: action done')
        _, state_dat = self.get_state()
        _, reward_dat = self.get_reward(tau)
        if self.need_jac:
            ind = np.random.randint(self.obs_siz)
            if self.mode == 0:
                mult = 0
            elif self.mode == 1:
                mult = 1
            else:
                mult = self.obs_siz
                if self.mode == 3:
                    self.act_lr = 1.0 / mult
            jac = calc_jac(self.state[ind], self.tau, free=False).reshape([1,-1]).numpy() * mult #* 0#self.obs_siz
            jacr = calc_jac(self.reward, self.tau, free=True).reshape([1,-1]).numpy()
            jac[np.isnan(jac)] = 0
            jacr[np.isnan(jacr)] = 0
            jac = np.clip(jac, -self.clip, self.clip) * self.act_lr
            jacr = np.clip(jacr, -self.clip, self.clip) * self.act_lr
        else:
            ind = 0
            jac = np.zeros([1, self.act_siz])
            jacr = np.zeros([1,self.act_siz])
    
        return self.state.detach().numpy(), self.reward.detach().numpy(), self.is_done(), {
            'jac': jac, 'jacr': jacr, 'jacind': np.array([ind]), 'forward_reward': reward_dat[0], 'ctrl_cost': reward_dat[1],
            'state_rot': state_dat[0], 'state_pos': state_dat[1], 'state_vel': state_dat[2]}#, 'state_joint': state_dat[3]}

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
                world = self.mb.body_to_world(i-1, v)
                out = [pd.Utils.getDouble(world[0]),pd.Utils.getDouble(world[1]),pd.Utils.getDouble(world[2])]
                f.write('v {} {} {}\n'.format(out[0],out[1],out[2]))
            for face in self.fcs[i]:
                f.write('f {} {} {}\n'.format(face[0],face[1],face[2]))

    def reset(self):
        if self.output_file is not None:
            print('GOOD!')
            self.vert = []
            self.fcs = []
            for i in range(len(self.mb.links)+1):
                self.addvert(i)
        self.q = self.init_q + (torch.rand_like(self.init_q) * 0.2 - 0.1) * self.random_init
        self.qd = self.init_qd + (torch.randn_like(self.init_qd) * 0.1) * self.random_init
        self.frame = 0
        self.reward = torch.tensor(0.)
        self.success = True
        self.xposbefore = api_diff.get_com(self.q, self.world)[3].detach() #self.q[4]
        return self.get_state()[0].numpy()

