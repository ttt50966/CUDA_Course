import os
import time

lattice_size = [32, 64, 128, 256, 512]

for i in range(len(lattice_size)):
    in_name = '{}_in'.format(lattice_size[i])
    cmd_name = 'cmd_{}'.format(lattice_size[i])
    with open(in_name,'wb') as f:
        f.write('{} {}'.format(lattice_size[i], lattice_size[i]))
    with open(cmd_name, 'wb') as f:
        cmd_content = ('Universe=vanilla\nExecutable=/opt/bin/runjob\nMachine_count=1\n\n'\
            'Output=condor.out\nError=condor.err\nLog=condor.log\n\n'\
            'Requirements=(MTYPE=="TEST")\n\nRequest_cpus=1\n\n'\
            'Initialdir=/work/d06222003/hw3/CPU\n\nArguments=./laplace_cpu {}_in {}_out\n\n'\
            'Notification=Always\nNotify_user=d06222003@g.ntu.edu.tw\n\n'\
            'Queue\n\n').format(lattice_size[i], lattice_size[i])
        f.write(cmd_content)
    os.system('condor_submit {}'.format(cmd_name))
    time.sleep(1)
