import os
import re
result_name = 'results.csv'
name = ["Enter the size (Nx, Ny) of the 2D lattice: ", "Enter the number of threads (tx,ty) per block:",
    "Input time for GPU:","total iterations (GPU) =",
    "Processing time for GPU:", "GPU Gflops:", "Output time for GPU:", "Total time for GPU:"]
lattice_size = [512]
block_size_x = [4, 8, 16, 32, 64, 128, 512]
block_size_y = [0]*7
for i in range(len(block_size_x)):
    block_size_y[i] = 1024 / block_size_x[i]
params = lattice_size , block_size_x, block_size_y
with open(result_name, 'wb') as f:
    f.write('Lattice size,Block size,Input time for GPU,total iterations (GPU),Processing time for GPU,GPU Gflops,Output time for GPU,Total time for GPU,\n')
    for i in range(len(lattice_size)):
        for j in range(len(block_size_x)):    
            out_name = '{}_{}_{}_out'.format(params[0][i], params[1][j], params[2][j])
            with open(out_name, 'r') as g:
                for line in g:         
                    for item in name:
                        strPosition = line.find(item, 0, len(line))
                        if strPosition!=-1:
                            str = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                            WriteLine = str[-1] + ","
                            f.write(WriteLine)
                            if item == name[-1]:
                                f.write('\n')
            g.close()

f.close()
