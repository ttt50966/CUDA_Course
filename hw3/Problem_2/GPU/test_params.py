import os
import time

lattice_size = [512]
block_size_x = [4, 8, 16, 32, 64, 128, 512]
block_size_y = [0]*7
for i in range(len(block_size_x)):
    block_size_y[i] = 1024 / block_size_x[i]
params = lattice_size , block_size_x, block_size_y

for i in range(len(lattice_size)):
    for j in range(len(block_size_x)):
        in_name = '{}_{}_{}_in'.format(params[0][i], params[1][j], params[2][j])
        cmd_name = 'cmd_{}_{}_{}'.format(params[0][i], params[1][j], params[2][j])
        with open(in_name,'wb') as f:
            f.write('{}\n{} {}\n{} {}\n{}'.format("GPU_ID", params[0][i], params[0][i], params[1][j], params[2][j], 1))
        with open(cmd_name, 'wb') as f:
            cmd_content = ('Universe=vanilla\nExecutable=/opt/bin/runjob\nMachine_count=1\n\n'
                'Output=condor.out\nError=condor.err\nLog=condor.log\n\n'
                'Requirements=(MTYPE=="TEST")\n\nRequest_cpus=1\n\n'
                'Initialdir=/work/d06222003/hw3/Problem_2/GPU\n\nArguments=./laplace {}_{}_{}_in {}_{}_{}_out\n\n'
                'Notification=Always\nNotify_user=d06222003@g.ntu.edu.tw\n\n'
                'Queue\n\n').format(
                params[0][i], params[1][j], params[2][j],
                params[0][i], params[1][j], params[2][j])
            f.write(cmd_content)
        os.system('condor_submit {}'.format(cmd_name))
        time.sleep(1)