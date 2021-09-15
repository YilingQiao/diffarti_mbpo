import numpy as np

def getin(R, h):
	H = h*2.
	pi=np.pi
	mcy=H*R*R*pi
	mhs=2*R*R*R*pi/3
	m=mcy+2*mhs
	ixx=mcy*(H*H/12+R*R/4)+2*mhs*(2*R*R/5+H*H/2+3*H*R/8)
	iyy=mcy*(R*R/2)+2*mhs*(2*R*R/5)
	izz=ixx
	print(m,ixx,iyy,izz)

getin(0.1,0.1)
