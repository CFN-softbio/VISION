import logging
import caproto

# from caproto._log import LogFormatter, color_log_format, log_date_format

# handler = logging.FileHandler('caproto_log')
# handler.setFormatter(
# LogFormatter(color_log_format, datefmt=log_date_format))
# log = logging.getLogger('caproto')
##log.setLevel('INFO')
# log.setLevel('DEBUG')
##log.setLevel('CRITICAL')
##log.disabled=True
# log.handlers.clear()
# log.addHandler(handler)


"""
set the evn viarable in COMMAND LINE
export OPHYD_CONTROL_LAYER=caproto

unset OPHYD_CONTROL_LAYER

without caproto, 0.5s exposure time
total time. by %time RE(count([pilatus2M]))

1st try: 3.8s
2---4 try: 2.5-2.6s


with caproto and 'INFO' level login, 0.5s exposure time
total time. by %time RE(count([pilatus2M]))

1st try: 7.32s
2---4 try: 6.8-7s
 
 
with caproto and 'DEBUG' level login, 0.5s exposure time
total time. by %time RE(count([pilatus2M]))

1st try: 9.86s
2---4 try: 9.1-9.3s

with caproto and 'CRITICAL' level login, 0.5s exposure time
total time. by %time RE(count([pilatus2M]))

1st try: 7.35s
2---4 try: 6.9-7.1s

with caproto and disabled login, 0.5s exposure time
total time. by %time RE(count([pilatus2M]))

1st try: 7.36s
2---4 try: 6.9-7.1s

===================test the exposure time without caproto=================
In [1]: %time RE(count([pilatus2M])) 
Transient Scan ID: 867308     Time: 2018/11/02 15:03:31
Persistent Unique Scan ID: '6d5f506f-883f-488a-b664-f88f0f188729'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:03:33.4 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['6d5f506f'] (scan num: 867308)



CPU times: user 992 ms, sys: 152 ms, total: 1.14 s
Wall time: 3.89 s
Out[1]: ('6d5f506f-883f-488a-b664-f88f0f188729',)

In [2]: %time RE(count([pilatus2M])) 
Transient Scan ID: 867309     Time: 2018/11/02 15:03:57
Persistent Unique Scan ID: '8864949f-9909-4cd5-942d-c9cf24fce2a8'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:03:58.9 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['8864949f'] (scan num: 867309)



CPU times: user 648 ms, sys: 96 ms, total: 744 ms
Wall time: 2.56 s
Out[2]: ('8864949f-9909-4cd5-942d-c9cf24fce2a8',)

In [3]: %time RE(count([pilatus2M])) 
Transient Scan ID: 867310     Time: 2018/11/02 15:04:02
Persistent Unique Scan ID: 'f9555465-5712-4308-8fef-fa3dda9f4752'
/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
  'Error:' + str(exc))
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:04:03.8 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['f9555465'] (scan num: 867310)



CPU times: user 740 ms, sys: 88 ms, total: 828 ms
Wall time: 2.6 s
Out[3]: ('f9555465-5712-4308-8fef-fa3dda9f4752',)

In [4]: %time RE(count([pilatus2M])) 
Transient Scan ID: 867311     Time: 2018/11/02 15:04:07
Persistent Unique Scan ID: 'fdae09c9-b9dd-4234-aefb-1299edffb4ec'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:04:09.4 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['fdae09c9'] (scan num: 867311)



CPU times: user 700 ms, sys: 112 ms, total: 812 ms
Wall time: 2.55 s
Out[4]: ('fdae09c9-b9dd-4234-aefb-1299edffb4ec',)

In [5]: %time RE(count([pilatus2M])) 
Transient Scan ID: 867312     Time: 2018/11/02 15:04:21
Persistent Unique Scan ID: '29a011e4-58d9-4acd-a516-cfd1c5c9c260'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:04:23.7 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['29a011e4'] (scan num: 867312)



CPU times: user 668 ms, sys: 108 ms, total: 776 ms
Wall time: 2.53 s
Out[5]: ('29a011e4-58d9-4acd-a516-cfd1c5c9c260',)


===================test the exposure time with caproto at 'INFO' level=================

In [1]: %time RE(count([pilatus2M]))
Transient Scan ID: 867313     Time: 2018/11/02 15:05:49
Persistent Unique Scan ID: '4224c7b8-23d6-4392-b21b-561f949939f4'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:05:51.9 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['4224c7b8'] (scan num: 867313)



CPU times: user 1.49 s, sys: 196 ms, total: 1.69 s
Wall time: 7.32 s
Out[1]: ('4224c7b8-23d6-4392-b21b-561f949939f4',)

In [2]: %time RE(count([pilatus2M]))
Transient Scan ID: 867314     Time: 2018/11/02 15:06:50
Persistent Unique Scan ID: '46a01cdb-9f8a-4006-991f-96b506b6b207'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:06:52.4 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['46a01cdb'] (scan num: 867314)



CPU times: user 1.29 s, sys: 76 ms, total: 1.37 s
Wall time: 7.04 s
Out[2]: ('46a01cdb-9f8a-4006-991f-96b506b6b207',)

In [3]: %time RE(count([pilatus2M]))
Transient Scan ID: 867315     Time: 2018/11/02 15:07:11
Persistent Unique Scan ID: '9fcf9f42-434d-4dbe-9504-4c3292c6666b'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:07:13.7 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['9fcf9f42'] (scan num: 867315)



^[[A
^[[ACPU times: user 1.13 s, sys: 68 ms, total: 1.2 s
Wall time: 6.8 s
Out[3]: ('9fcf9f42-434d-4dbe-9504-4c3292c6666b',)

In [4]: %time RE(count([pilatus2M]))

Transient Scan ID: 867316     Time: 2018/11/02 15:07:18
Persistent Unique Scan ID: '7d91aa6c-de37-4a55-8a26-7710cafe9e5a'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:07:20.7 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['7d91aa6c'] (scan num: 867316)



CPU times: user 1.13 s, sys: 124 ms, total: 1.25 s
Wall time: 7.03 s
Out[4]: ('7d91aa6c-de37-4a55-8a26-7710cafe9e5a',)

In [5]: 

In [5]: %time RE(count([pilatus2M]))
Transient Scan ID: 867317     Time: 2018/11/02 15:07:27
Persistent Unique Scan ID: '83a615b1-d948-4ad7-a434-c4e772f8b37b'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:07:29.8 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['83a615b1'] (scan num: 867317)



CPU times: user 1.26 s, sys: 100 ms, total: 1.36 s
Wall time: 7.01 s
Out[5]: ('83a615b1-d948-4ad7-a434-c4e772f8b37b',)


===================test the exposure time with caproto at 'DEBUG' level=================
In [1]: log.setLevel('DEBUG')

In [2]: %time RE(count([pilatus2M]))
^[[A
Transient Scan ID: 867318     Time: 2018/11/02 15:08:47
Persistent Unique Scan ID: '6848fe0a-d71d-4a05-9e8e-b0157f8e2f02'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:08:50.7 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['6848fe0a'] (scan num: 867318)



CPU times: user 3.93 s, sys: 560 ms, total: 4.49 s
Wall time: 9.86 s
Out[2]: ('6848fe0a-d71d-4a05-9e8e-b0157f8e2f02',)

In [3]: %time RE(count([pilatus2M]))
Transient Scan ID: 867319     Time: 2018/11/02 15:08:56
Persistent Unique Scan ID: 'fb239da4-9e9d-4431-b177-cd00af54e679'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:08:59.8 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['fb239da4'] (scan num: 867319)



CPU times: user 3.42 s, sys: 332 ms, total: 3.75 s
Wall time: 9.03 s
Out[3]: ('fb239da4-9e9d-4431-b177-cd00af54e679',)

In [4]: %time RE(count([pilatus2M]))
^[[A
Transient Scan ID: 867320     Time: 2018/11/02 15:09:29
Persistent Unique Scan ID: 'ee2f0e3b-f792-4510-83bb-9007fb4555ec'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:09:32.5 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['ee2f0e3b'] (scan num: 867320)



CPU times: user 3.66 s, sys: 356 ms, total: 4.01 s
Wall time: 9.35 s
Out[4]: ('ee2f0e3b-f792-4510-83bb-9007fb4555ec',)

In [5]: %time RE(count([pilatus2M]))
Transient Scan ID: 867321     Time: 2018/11/02 15:09:38
Persistent Unique Scan ID: 'fa008494-5b6b-4f66-a7b9-2d8863f76b14'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:09:41.8 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['fa008494'] (scan num: 867321)



CPU times: user 3.59 s, sys: 340 ms, total: 3.93 s
Wall time: 9.28 s
Out[5]: ('fa008494-5b6b-4f66-a7b9-2d8863f76b14',)

In [6]: %time RE(count([pilatus2M]))
Transient Scan ID: 867322     Time: 2018/11/02 15:10:00
Persistent Unique Scan ID: '8b129720-4da7-4c84-9a2d-41fab456dadf'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:10:03.9 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['8b129720'] (scan num: 867322)



CPU times: user 3.44 s, sys: 388 ms, total: 3.82 s
Wall time: 9.13 s
Out[6]: ('8b129720-4da7-4c84-9a2d-41fab456dadf',)

In [7]: %time RE(count([pilatus2M]))
Transient Scan ID: 867323     Time: 2018/11/02 15:10:11
Persistent Unique Scan ID: '33101762-fc9b-4efa-8d8c-e55669924fbc'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:10:14.7 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['33101762'] (scan num: 867323)



CPU times: user 3.46 s, sys: 368 ms, total: 3.83 s
Wall time: 9.16 s
Out[7]: ('33101762-fc9b-4efa-8d8c-e55669924fbc',)


===================test the exposure time with caproto at 'CRITICAL' level=================


In [1]: log.setLevel('CRITICAL')

In [2]: %time RE(count([pilatus2M]))
Transient Scan ID: 867324     Time: 2018/11/02 15:11:27
Persistent Unique Scan ID: 'a0930f36-1127-49d4-8b98-5549341a313f'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:11:29.3 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['a0930f36'] (scan num: 867324)



CPU times: user 1.39 s, sys: 148 ms, total: 1.54 s
Wall time: 7.35 s
Out[2]: ('a0930f36-1127-49d4-8b98-5549341a313f',)

In [3]: %time RE(count([pilatus2M]))
Transient Scan ID: 867325     Time: 2018/11/02 15:11:37
Persistent Unique Scan ID: '8bb969e2-657d-4c10-b217-c2acd5e6d348'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
                                                                                                   
New stream: 'primary'
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:11:39.1 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['8bb969e2'] (scan num: 867325)



CPU times: user 1.33 s, sys: 100 ms, total: 1.43 s
Wall time: 7.05 s
Out[3]: ('8bb969e2-657d-4c10-b217-c2acd5e6d348',)

In [4]: %time RE(count([pilatus2M]))
Transient Scan ID: 867326     Time: 2018/11/02 15:11:44
Persistent Unique Scan ID: 'a3411531-b8a7-4a4c-be1b-38032cf42c80'
/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
  'Error:' + str(exc))
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:11:46.2 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['a3411531'] (scan num: 867326)



CPU times: user 1.21 s, sys: 124 ms, total: 1.34 s
Wall time: 6.97 s
Out[4]: ('a3411531-b8a7-4a4c-be1b-38032cf42c80',)

In [5]: %time RE(count([pilatus2M]))
Transient Scan ID: 867327     Time: 2018/11/02 15:11:56
Persistent Unique Scan ID: '03c919fa-9af9-4827-9de6-a3a6dc992b8a'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:11:58.0 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['03c919fa'] (scan num: 867327)



CPU times: user 1.26 s, sys: 100 ms, total: 1.36 s
Wall time: 6.9 s
Out[5]: ('03c919fa-9af9-4827-9de6-a3a6dc992b8a',)

In [6]: %time RE(count([pilatus2M]))
Transient Scan ID: 867328     Time: 2018/11/02 15:12:08
Persistent Unique Scan ID: '5a816fe6-8876-4249-811d-b6f125e899c0'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:12:10.1 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['5a816fe6'] (scan num: 867328)



CPU times: user 1.33 s, sys: 92 ms, total: 1.42 s
Wall time: 7.06 s
Out[6]: ('5a816fe6-8876-4249-811d-b6f125e899c0',)

In [7]: %time RE(count([pilatus2M]))
Transient Scan ID: 867329     Time: 2018/11/02 15:12:19
Persistent Unique Scan ID: '55d1387e-2f6a-4d9c-a19c-437999fa40a3'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:12:21.5 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['55d1387e'] (scan num: 867329)



CPU times: user 1.27 s, sys: 52 ms, total: 1.32 s
Wall time: 6.88 s
Out[7]: ('55d1387e-2f6a-4d9c-a19c-437999fa40a3',)


===================test the exposure time with caproto at  log.disabled=True =================
In [1]: log.disabled=True

In [2]: %time RE(count([pilatus2M]))
^[[A
Transient Scan ID: 867330     Time: 2018/11/02 15:13:32
Persistent Unique Scan ID: 'b219651b-2072-43a8-b2ee-3ee4babea253'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:13:34.4 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['b219651b'] (scan num: 867330)



CPU times: user 1.33 s, sys: 212 ms, total: 1.54 s
Wall time: 7.36 s
Out[2]: ('b219651b-2072-43a8-b2ee-3ee4babea253',)

In [3]: %time RE(count([pilatus2M]))
Transient Scan ID: 867331     Time: 2018/11/02 15:13:39
Persistent Unique Scan ID: 'f979ff47-b4ec-46e1-bea8-d8776339c605'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:13:41.5 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['f979ff47'] (scan num: 867331)



CPU times: user 1.25 s, sys: 124 ms, total: 1.37 s
Wall time: 7.1 s
Out[3]: ('f979ff47-b4ec-46e1-bea8-d8776339c605',)

In [4]: %time RE(count([pilatus2M]))
Transient Scan ID: 867332     Time: 2018/11/02 15:14:11
Persistent Unique Scan ID: 'd259a610-7edb-4bc6-b901-38b0deaf6e5e'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:14:13.8 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['d259a610'] (scan num: 867332)



CPU times: user 1.18 s, sys: 104 ms, total: 1.29 s
Wall time: 6.9 s
Out[4]: ('d259a610-7edb-4bc6-b901-38b0deaf6e5e',)

In [5]: %time RE(count([pilatus2M]))
Transient Scan ID: 867333     Time: 2018/11/02 15:14:33
Persistent Unique Scan ID: 'da6b9b84-2587-46ed-bc68-f45812ed70a9'

/opt/conda_envs/collection-2018-3.2/lib/python3.6/site-packages/nslsii/__init__.py:262: UserWarning: This olog is giving errors. This will not be logged.Error:0
New stream: 'primary'                                                                              
+-----------+------------+------------------------+------------------------+
|   seq_num |       time | pilatus2M_stats3_total | pilatus2M_stats4_total |
+-----------+------------+------------------------+------------------------+
|         1 | 15:14:35.6 |                      0 |                      0 |
+-----------+------------+------------------------+------------------------+
generator count ['da6b9b84'] (scan num: 867333)



CPU times: user 1.31 s, sys: 132 ms, total: 1.44 s
Wall time: 7.06 s
Out[5]: ('da6b9b84-2587-46ed-bc68-f45812ed70a9',)


"""
