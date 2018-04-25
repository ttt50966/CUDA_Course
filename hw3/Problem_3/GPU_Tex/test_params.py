import os
import time

lattice_size = [32, 64, 128]
block_size = [2, 4, 8]
params = lattice_size , block_size

for i in range(len(lattice_size)):
    for j in range(len(block_size)):
        in_name = '{}_{}_in'.format(params[0][i], params[1][j])
        cmd_name = 'cmd_{}_{}'.format(params[0][i], params[1][j])
        with open(in_name,'wb') as f:
            f.write('{}\n{} {} {}\n{} {} {}\n{}'.format("GPU_ID", params[0][i], params[0][i], params[0][i], params[1][j], params[1][j], params[1][j], 0))
        with open(cmd_name, 'wb') as f:
            cmd_content = ('Universe=vanilla\nExecutable=/opt/bin/runjob\nMachine_count=1\n\n'
                'Output=condor.out\nError=condor.err\nLog=condor.log\n\n'
                'Requirements=(MTYPE=="TEST")\n\nRequest_cpus=1\n\n'
                'Initialdir=/work/d06222003/hw3/Problem_3/GPU_Tex\n\nArguments=./laplaceTex {}_{}_in {}_{}_out\n\n'
                'Notification=Always\nNotify_user=d06222003@g.ntu.edu.tw\n\n'
                'Queue\n\n').format(
                params[0][i], params[1][j],
                params[0][i], params[1][j])
            f.write(cmd_content)
        os.system('condor_submit {}'.format(cmd_name))
        time.sleep(1)