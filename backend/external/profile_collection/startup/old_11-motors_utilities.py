# def wh_all():
# wh_pos([mono_bragg,mono_pitch2,mono_roll2,mono_perp2])
# wh_pos([mir_usx,mir_dsx,mir_usy,mir_dsyi,mir_dsyo,mir_bend])
# wh_pos(s0)
# wh_pos(s1)
# wh_pos(s2)
# wh_pos(s3)
# wh_pos(s4)
# wh_pos(s5)
# wh_pos([bim3y,fs3y,bim4y,bim5y])
# wh_pos([smx,smy,sth,schi,sphi,srot,strans,strans2,stilt,stilt2])
# wh_pos([camx,camy,cam2x,cam2z])
# wh_pos([DETx,DETy,WAXSx,SAXSx,SAXSy])
# wh_pos([bsx,bsy,bsphi])
# wh_pos([armz,armx,armphi,army,armr])


# def wh_offsets():
# print('Direction: 0--Pos, 1--Neg\n')

### mono
# wh_pos([mono_bragg,mono_pitch2,mono_roll2,mono_perp2])
# print('mono_bragg:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mono:DMM-Ax:Bragg}Mtr.OFF'),caget('XF:11BMA-OP{Mono:DMM-Ax:Bragg}Mtr.DIR')))
# print('mono_pitch2:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mono:DMM-Ax:P2}Mtr.OFF'),caget('XF:11BMA-OP{Mono:DMM-Ax:P2}Mtr.DIR')))
# print('mono_roll2:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mono:DMM-Ax:R2}Mtr.OFF'),caget('XF:11BMA-OP{Mono:DMM-Ax:R2}Mtr.DIR')))
# print('mono_perp2:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mono:DMM-Ax:Y2}Mtr.OFF'),caget('XF:11BMA-OP{Mono:DMM-Ax:Y2}Mtr.DIR')))

### mirror
# wh_pos([mir_usx,mir_dsx,mir_usy,mir_dsyi,mir_dsyo,mir_bend])
# print('mir_usx:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:XU}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:XU}Mtr.DIR')))
# print('mir_dsx:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:XD}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:XD}Mtr.DIR')))
# print('mir_usy:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:YU}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:YU}Mtr.DIR')))
# print('mir_dsyi:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:YDI}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:YDI}Mtr.DIR')))
# print('mir_dsyo:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:YDO}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:YDO}Mtr.DIR')))
# print('mir_bend:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:UB}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:UB}Mtr.DIR')))

### slits S0
# wh_pos(s0)
# print('s0.tp:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Slt:0-Ax:T}Mtr.OFF'),caget('XF:11BMA-OP{Slt:0-Ax:T}Mtr.DIR')))
# print('s0.bt:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Slt:0-Ax:B}Mtr.OFF'),caget('XF:11BMA-OP{Slt:0-Ax:B}Mtr.DIR')))
# print('s0.ob:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Slt:0-Ax:O}Mtr.OFF'),caget('XF:11BMA-OP{Slt:0-Ax:O}Mtr.DIR')))
# print('s0.ib:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Slt:0-Ax:I}Mtr.OFF'),caget('XF:11BMA-OP{Slt:0-Ax:I}Mtr.DIR')))

### slits S1
# wh_pos(s1)
# print('s1.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:1-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:1-Ax:XC}Mtr.DIR')))
# print('s1.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:1-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:1-Ax:XG}Mtr.DIR')))
# print('s1.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:1-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:1-Ax:YC}Mtr.DIR')))
# print('s1.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:1-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:1-Ax:YG}Mtr.DIR')))

### slits S2
# wh_pos(s2)
# print('s2.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:2-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:2-Ax:XC}Mtr.DIR')))
# print('s2.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:2-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:2-Ax:XG}Mtr.DIR')))
# print('s2.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:2-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:2-Ax:YC}Mtr.DIR')))
# print('s2.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:2-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:2-Ax:YG}Mtr.DIR')))

### slits S3
# wh_pos(s3)
# print('s3.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:3-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:3-Ax:XC}Mtr.DIR')))
# print('s3.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:3-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:3-Ax:XG}Mtr.DIR')))
# print('s3.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:3-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:3-Ax:YC}Mtr.DIR')))
# print('s3.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:3-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:3-Ax:YG}Mtr.DIR')))

### slits S4
# wh_pos(s4)
# print('s4.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:4-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:4-Ax:XC}Mtr.DIR')))
# print('s4.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:4-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:4-Ax:XG}Mtr.DIR')))
# print('s4.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:4-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:4-Ax:YC}Mtr.DIR')))
# print('s4.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:4-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:4-Ax:YG}Mtr.DIR')))

### slits S5
# wh_pos(s5)
# print('s5.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:5-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:5-Ax:XC}Mtr.DIR')))
# print('s5.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:5-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:5-Ax:XG}Mtr.DIR')))
# print('s5.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:5-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:5-Ax:YC}Mtr.DIR')))
# print('s5.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:5-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:5-Ax:YG}Mtr.DIR')))


### diagnostic stages
# wh_pos([bim3y,fs3y,bim4y,bim5y])
# print('bim3y:  offset = %f, direction = %d' % (caget('XF:11BMB-BI{IM:3-Ax:Y}Mtr.OFF'),caget('XF:11BMB-BI{IM:3-Ax:Y}Mtr.DIR')))
# print('fs3y:  offset = %f, direction = %d' % (caget('XF:11BMB-BI{FS:3-Ax:Y}Mtr.OFF'),caget('XF:11BMB-BI{FS:3-Ax:Y}Mtr.DIR')))
# print('bim4y:  offset = %f, direction = %d' % (caget('XF:11BMB-BI{IM:4-Ax:Y}Mtr.OFF'),caget('XF:11BMB-BI{IM:4-Ax:Y}Mtr.DIR')))
# print('bim5y:  offset = %f, direction = %d' % (caget('XF:11BMB-BI{IM:5-Ax:Y}Mtr.OFF'),caget('XF:11BMB-BI{IM:5-Ax:Y}Mtr.DIR')))


### sample stages
# wh_pos([smx,smy,sth,schi,sphi,srot,strans])
# print('smx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr.DIR')))
# print('smy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:Z}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:Z}Mtr.DIR')))
# print('sth:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:theta}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:theta}Mtr.DIR')))
# print('schi:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:chi}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:chi}Mtr.DIR')))
# print('sphi:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:phi}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:phi}Mtr.DIR')))
# print('srot:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Srot}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Srot}Mtr.DIR')))
# print('strans:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Strans}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Strans}Mtr.DIR')))
# print('strans2:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Strans2}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Strans2}Mtr.DIR')))
# print('stilt:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Stilt}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Stilt}Mtr.DIR')))
# print('stilt2:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Stilt2}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Stilt2}Mtr.DIR')))


### camera
# wh_pos([camx,camy])
# print('camx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Cam:OnAxis-Ax:X1}Mtr.OFF'),caget('XF:11BMB-ES{Cam:OnAxis-Ax:X1}Mtr.DIR')))
# print('camy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Cam:OnAxis-Ax:Y1}Mtr.OFF'),caget('XF:11BMB-ES{Cam:OnAxis-Ax:Y1}Mtr.OFF')))
# print('cam2x:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Cam:OnAxis-Ax:X2}Mtr.OFF'),caget('XF:11BMB-ES{Cam:OnAxis-Ax:X2}Mtr.DIR')))
# print('cam2z:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Cam:OnAxis-Ax:Y2}Mtr.OFF'),caget('XF:11BMB-ES{Cam:OnAxis-Ax:Y2}Mtr.OFF')))


### detector stages
# wh_pos([DETx,DETy,WAXSx,SAXSx,SAXSy])
# print('DETx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:Stg-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{Det:Stg-Ax:X}Mtr.DIR')))
# print('DETy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:Stg-Ax:Y}Mtr.OFF'),caget('XF:11BMB-ES{Det:Stg-Ax:Y}Mtr.DIR')))
# print('WAXSx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:WAXS-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{Det:WAXS-Ax:X}Mtr.DIR')))
# print('SAXSx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:SAXS-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{Det:SAXS-Ax:X}Mtr.DIR')))
# print('SAXSy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:SAXS-Ax:Y}Mtr.OFF'),caget('XF:11BMB-ES{Det:SAXS-Ax:Y}Mtr.DIR')))


### beamstop
# wh_pos([bsx,bsy,bsphi])
# print('bsx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{BS:SAXS-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{BS:SAXS-Ax:X}Mtr.DIR')))
# print('bsy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{BS:SAXS-Ax:Y}Mtr.OFF'),caget('XF:11BMB-ES{BS:SAXS-Ax:Y}Mtr.DIR')))
# print('bsphi:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{BS:SAXS-Ax:phi}Mtr.OFF'),caget('XF:11BMB-ES{BS:SAXS-Ax:phi}Mtr.DIR')))


### sample exchanger
# wh_pos([armz,armx,armphi,army,armr])
# print('armz:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Z}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Z}Mtr.DIR')))
# print('armx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:X}Mtr.DIR')))
# print('armphi:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Yaw}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Yaw}Mtr.DIR')))
# print('army:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Y}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Y}Mtr.DIR')))
# print('armr:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:ArmR}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:ArmR}Mtr.DIR')))


# def wh_all():
# %wa([mono_bragg,mono_pitch2,mono_roll2,mono_perp2])
# %wa([mir_usx,mir_dsx,mir_usy,mir_dsyi,mir_dsyo,mir_bend])
# %wa(s0)
# %wa(s1)
# %wa(s2)
# %wa(s3)
# %wa(s4)
# %wa(s5)
# %wa([bim3y,fs3y,bim4y,bim5y])
# %wa([smx,smy,sth,schi,sphi,srot,strans,strans2,stilt,stilt2])
# %wa([camx,camy,cam2x,cam2z])
# %wa([DETx,DETy,WAXSx,SAXSx,SAXSy])
# %wa([bsx,bsy,bsphi])
# %wa([armz,armx,armphi,army,armr])


# def wh_offsets():
# print('Direction: 0--Pos, 1--Neg\n')

### mono
# %wa([mono_bragg,mono_pitch2,mono_roll2,mono_perp2])
# print('mono_bragg:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mono:DMM-Ax:Bragg}Mtr.OFF'),caget('XF:11BMA-OP{Mono:DMM-Ax:Bragg}Mtr.DIR')))
# print('mono_pitch2:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mono:DMM-Ax:P2}Mtr.OFF'),caget('XF:11BMA-OP{Mono:DMM-Ax:P2}Mtr.DIR')))
# print('mono_roll2:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mono:DMM-Ax:R2}Mtr.OFF'),caget('XF:11BMA-OP{Mono:DMM-Ax:R2}Mtr.DIR')))
# print('mono_perp2:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mono:DMM-Ax:Y2}Mtr.OFF'),caget('XF:11BMA-OP{Mono:DMM-Ax:Y2}Mtr.DIR')))

### mirror
# %wa([mir_usx,mir_dsx,mir_usy,mir_dsyi,mir_dsyo,mir_bend])
# print('mir_usx:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:XU}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:XU}Mtr.DIR')))
# print('mir_dsx:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:XD}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:XD}Mtr.DIR')))
# print('mir_usy:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:YU}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:YU}Mtr.DIR')))
# print('mir_dsyi:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:YDI}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:YDI}Mtr.DIR')))
# print('mir_dsyo:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:YDO}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:YDO}Mtr.DIR')))
# print('mir_bend:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Mir:Tor-Ax:UB}Mtr.OFF'),caget('XF:11BMA-OP{Mir:Tor-Ax:UB}Mtr.DIR')))

### slits S0
# %wa(s0)
# print('s0.tp:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Slt:0-Ax:T}Mtr.OFF'),caget('XF:11BMA-OP{Slt:0-Ax:T}Mtr.DIR')))
# print('s0.bt:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Slt:0-Ax:B}Mtr.OFF'),caget('XF:11BMA-OP{Slt:0-Ax:B}Mtr.DIR')))
# print('s0.ob:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Slt:0-Ax:O}Mtr.OFF'),caget('XF:11BMA-OP{Slt:0-Ax:O}Mtr.DIR')))
# print('s0.ib:  offset = %f, direction = %d' % (caget('XF:11BMA-OP{Slt:0-Ax:I}Mtr.OFF'),caget('XF:11BMA-OP{Slt:0-Ax:I}Mtr.DIR')))

### slits S1
# %wa(s1)
# print('s1.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:1-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:1-Ax:XC}Mtr.DIR')))
# print('s1.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:1-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:1-Ax:XG}Mtr.DIR')))
# print('s1.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:1-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:1-Ax:YC}Mtr.DIR')))
# print('s1.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:1-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:1-Ax:YG}Mtr.DIR')))

### slits S2
# %wa(s2)
# print('s2.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:2-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:2-Ax:XC}Mtr.DIR')))
# print('s2.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:2-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:2-Ax:XG}Mtr.DIR')))
# print('s2.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:2-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:2-Ax:YC}Mtr.DIR')))
# print('s2.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:2-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:2-Ax:YG}Mtr.DIR')))

### slits S3
# %wa(s3)
# print('s3.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:3-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:3-Ax:XC}Mtr.DIR')))
# print('s3.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:3-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:3-Ax:XG}Mtr.DIR')))
# print('s3.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:3-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:3-Ax:YC}Mtr.DIR')))
# print('s3.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:3-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:3-Ax:YG}Mtr.DIR')))

### slits S4
# %wa(s4)
# print('s4.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:4-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:4-Ax:XC}Mtr.DIR')))
# print('s4.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:4-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:4-Ax:XG}Mtr.DIR')))
# print('s4.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:4-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:4-Ax:YC}Mtr.DIR')))
# print('s4.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:4-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:4-Ax:YG}Mtr.DIR')))

### slits S5
# %wa(s5)
# print('s5.xc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:5-Ax:XC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:5-Ax:XC}Mtr.DIR')))
# print('s5.xg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:5-Ax:XG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:5-Ax:XG}Mtr.DIR')))
# print('s5.yc:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:5-Ax:YC}Mtr.OFF'),caget('XF:11BMB-OP{Slt:5-Ax:YC}Mtr.DIR')))
# print('s5.yg:  offset = %f, direction = %d' % (caget('XF:11BMB-OP{Slt:5-Ax:YG}Mtr.OFF'),caget('XF:11BMB-OP{Slt:5-Ax:YG}Mtr.DIR')))


### diagnostic stages
# %wa([bim3y,fs3y,bim4y,bim5y])
# print('bim3y:  offset = %f, direction = %d' % (caget('XF:11BMB-BI{IM:3-Ax:Y}Mtr.OFF'),caget('XF:11BMB-BI{IM:3-Ax:Y}Mtr.DIR')))
# print('fs3y:  offset = %f, direction = %d' % (caget('XF:11BMB-BI{FS:3-Ax:Y}Mtr.OFF'),caget('XF:11BMB-BI{FS:3-Ax:Y}Mtr.DIR')))
# print('bim4y:  offset = %f, direction = %d' % (caget('XF:11BMB-BI{IM:4-Ax:Y}Mtr.OFF'),caget('XF:11BMB-BI{IM:4-Ax:Y}Mtr.DIR')))
# print('bim5y:  offset = %f, direction = %d' % (caget('XF:11BMB-BI{IM:5-Ax:Y}Mtr.OFF'),caget('XF:11BMB-BI{IM:5-Ax:Y}Mtr.DIR')))


### sample stages
# %wa([smx,smy,sth,schi,sphi,srot,strans])
# print('smx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr.DIR')))
# print('smy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:Z}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:Z}Mtr.DIR')))
# print('sth:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:theta}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:theta}Mtr.DIR')))
# print('schi:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:chi}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:chi}Mtr.DIR')))
# print('sphi:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Chm:Smpl-Ax:phi}Mtr.OFF'),caget('XF:11BMB-ES{Chm:Smpl-Ax:phi}Mtr.DIR')))
# print('srot:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Srot}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Srot}Mtr.DIR')))
# print('strans:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Strans}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Strans}Mtr.DIR')))
# print('strans2:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Strans2}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Strans2}Mtr.DIR')))
# print('stilt:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Stilt}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Stilt}Mtr.DIR')))
# print('stilt2:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Stilt2}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Stilt2}Mtr.DIR')))


### camera
# %wa([camx,camy])
# print('camx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Cam:OnAxis-Ax:X1}Mtr.OFF'),caget('XF:11BMB-ES{Cam:OnAxis-Ax:X1}Mtr.DIR')))
# print('camy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Cam:OnAxis-Ax:Y1}Mtr.OFF'),caget('XF:11BMB-ES{Cam:OnAxis-Ax:Y1}Mtr.OFF')))
# print('cam2x:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Cam:OnAxis-Ax:X2}Mtr.OFF'),caget('XF:11BMB-ES{Cam:OnAxis-Ax:X2}Mtr.DIR')))
# print('cam2z:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Cam:OnAxis-Ax:Y2}Mtr.OFF'),caget('XF:11BMB-ES{Cam:OnAxis-Ax:Y2}Mtr.OFF')))


### detector stages
# %wa([DETx,DETy,WAXSx,SAXSx,SAXSy])
# print('DETx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:Stg-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{Det:Stg-Ax:X}Mtr.DIR')))
# print('DETy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:Stg-Ax:Y}Mtr.OFF'),caget('XF:11BMB-ES{Det:Stg-Ax:Y}Mtr.DIR')))
# print('WAXSx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:WAXS-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{Det:WAXS-Ax:X}Mtr.DIR')))
# print('SAXSx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:SAXS-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{Det:SAXS-Ax:X}Mtr.DIR')))
# print('SAXSy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{Det:SAXS-Ax:Y}Mtr.OFF'),caget('XF:11BMB-ES{Det:SAXS-Ax:Y}Mtr.DIR')))


### beamstop
# %wa([bsx,bsy,bsphi])
# print('bsx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{BS:SAXS-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{BS:SAXS-Ax:X}Mtr.DIR')))
# print('bsy:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{BS:SAXS-Ax:Y}Mtr.OFF'),caget('XF:11BMB-ES{BS:SAXS-Ax:Y}Mtr.DIR')))
# print('bsphi:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{BS:SAXS-Ax:phi}Mtr.OFF'),caget('XF:11BMB-ES{BS:SAXS-Ax:phi}Mtr.DIR')))


### sample exchanger
# %wa([armz,armx,armphi,army,armr])
# print('armz:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Z}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Z}Mtr.DIR')))
# print('armx:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:X}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:X}Mtr.DIR')))
# print('armphi:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Yaw}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Yaw}Mtr.DIR')))
# print('army:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:Y}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:Y}Mtr.DIR')))
# print('armr:  offset = %f, direction = %d' % (caget('XF:11BMB-ES{SM:1-Ax:ArmR}Mtr.OFF'),caget('XF:11BMB-ES{SM:1-Ax:ArmR}Mtr.DIR')))
