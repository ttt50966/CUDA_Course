// Solve the Laplace equation on a 2D lattice with boundary conditions.
//
// compile with the following command:
//
// (for GTX970)
// nvcc -arch=compute_52 -code=sm_52,sm_52 -O3 -m64 -o laplace laplace.cu
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o laplace laplace.cu


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// field variables
float* h_new;   // host field vectors
float* h_old;   

int     MAX=1000000;          // maximum iterations
double  eps=1.0e-10;          // stopping criterion

int main(void)
{
    printf("Solve Laplace equation on a 3D lattice with boundary conditions\n");

    int Nx,Ny,Nz;    // lattice size

    printf("Enter the size of the cubic lattice: ");
    scanf("%d %d %d",&Nx,&Ny,&Nz);        
    printf("%d %d %d\n",Nx,Ny,Nz);        

    int size = Nx*Ny*Nz*sizeof(float); 
    h_new = (float*)malloc(size);
    h_old = (float*)malloc(size);

    memset(h_old, 0, size);
    memset(h_new, 0, size);

//    for(int j=0;j<Ny;j++) 
//    for(int i=0;i<Nx;i++) 
//      h_new[i+j*Nx]=0.0;

    // Initialize the field vector with boundary conditions
    for(int y=0; y<Ny; y++) {
      for(int z=0; z<Nz; z++){
        h_new[0+Nx*(y)+Nx*Ny*(z)]=0.0;  
        h_old[0+Nx*(y)+Nx*Ny*(z)]=0.0;  
        h_new[Nx-1+Nx*(y)+Nx*Ny*(z)]=0.0;  
        h_old[Nx-1+Nx*(y)+Nx*Ny*(z)]=0.0;  
      }
    }
    for(int z=0; z<Nz; z++) {
      for(int x=0; x<Nx; x++){
        h_new[x+Nx*(0)+Nx*Ny*(z)]=0.0;  
        h_old[x+Nx*(0)+Nx*Ny*(z)]=0.0;  
        h_new[x+Nx*(Ny-1)+Nx*Ny*(z)]=0.0;  
        h_old[x+Nx*(Ny-1)+Nx*Ny*(z)]=0.0;  
      }
    }  
    for(int x=0; x<Nx; x++) {
      for(int y=0; y<Ny; y++){
        h_new[x+Nx*(y)+Nx*Ny*(Nz-1)]=1.0;  
        h_old[x+Nx*(y)+Nx*Ny*(Nz-1)]=1.0;  
        h_new[x+Nx*(y)+Nx*Ny*(0)]=0.0;  
        h_old[x+Nx*(y)+Nx*Ny*(0)]=0.0;  
      }
    }
      
    

    FILE *out1;          // save initial configuration in phi_initial_Tex.dat
    out1 = fopen("phi_initial.dat","w");

    fprintf(out1, "Inital field configuration:\n");
    for(int k=Nz-1;k>-1;k--){
      for(int j=Ny-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          fprintf(out1,"%.2e ",h_new[i+j*Nx+k*Nx*Ny]);
        }
        fprintf(out1,"\n");
      }
      fprintf(out1,"\n");
      fprintf(out1,"\n");
    }
    fclose(out1);

    printf("\n");
    printf("Inital field configuration:\n");
    for(int k=Nz-1;k>-1;k--){
      for(int j=Ny-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          printf("%.2e ",h_new[i+j*Nx+k*Nx*Ny]);
        }
        printf("\n");
      }
      printf("\n");
      printf("\n");
    }
    printf("\n");

    // create the timer
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //start the timer
    cudaEventRecord(start,0);

    // to compute the reference solution

    double error = 10*eps;  // any value bigger eps is OK 
    int iter = 0;           // counter for iterations

    volatile bool flag = true;     

    float t, l, r, b, u, d;     // top, left, right, bottom
    double diff; 
    int site, ym1, xm1, xp1, yp1, zm1, zp1;

    while ( (error > eps) && (iter < MAX) ) {
      if(flag) {
        error = 0.0;
        for(int z=0; z<Nz; z++) {
        for(int y=0; y<Ny; y++) {
        for(int x=0; x<Nx; x++) {
          if(x==0 || x==Nx-1 || y==0 || y==Ny-1 || z==0 || z==Nz-1) {   
          }
          else {
            site = x+y*Nx + z*Nx*Ny;
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = site - Nx;   // y-1
            yp1 = site + Nx;   // y+1
            zm1 = site + Nx*Ny;
            zp1 = site - Nx*Ny;
            b = h_old[ym1]; 
            l = h_old[xm1]; 
            r = h_old[xp1]; 
            t = h_old[yp1]; 
            u = h_old[zm1];
            d = h_old[zp1];
            h_new[site] = (b+l+r+t+u+d)/6;
            diff = h_new[site]-h_old[site]; 
            error = error + diff*diff;
          }
        } 
        } 
        }
      }
      else {
        error = 0.0;
        for(int z=0; z<Nz; z++) {
        for(int y=0; y<Ny; y++) {
        for(int x=0; x<Nx; x++) { 
          if(x==0 || x==Nx-1 || y==0 || y==Ny-1 || z==0 || z==Nz-1) {
          }
          else {
            site = x+y*Nx + z*Nx*Ny;
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = site - Nx;   // y-1
            yp1 = site + Nx;   // y+1
            zm1 = site + Nx*Ny;
            zp1 = site - Nx*Ny;
            b = h_new[ym1]; 
            l = h_new[xm1]; 
            r = h_new[xp1]; 
            t = h_new[yp1]; 
            u = h_new[zm1];
            d = h_new[zp1];
            h_old[site] = (b+l+r+t+u+d)/6;
            diff = h_new[site]-h_old[site]; 
            error = error + diff*diff;
          } 
        }
        }
        }
      }
      flag = !flag;
      iter++;
      error = sqrt(error);

//      printf("error = %.15e\n",error);
//      printf("iteration = %d\n",iter);

    }         // exit if error < eps

    printf("error = %.15e\n",error);
    printf("total iterations = %d\n",iter);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    double flops = 9.0*(Nx-2)*(Ny-2)*(Nz-2)*iter; 
    printf("CPU Gflops: %lf\n",flops/(1000000.0*cputime));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    FILE *outc;          // save final configuration in phi_CPU.dat
    outc = fopen("phi_CPU.dat","w");

    fprintf(outc,"Final field configuration (CPU):\n");
    for(int k = Nz-1; k>-1;k--){
      for(int j=Ny-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          fprintf(outc,"%.2e ",h_new[i+j*Nx+k*Nx*Ny]);
        }
        fprintf(outc,"\n");
      }
      fprintf(outc,"\n");
      fprintf(outc,"\n");
    }
    fclose(outc);

    printf("\n");
    printf("Final field configuration (CPU):\n");
    for(int k=Nz-1;k>-1;k--){
      for(int j=Ny-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          printf("%.2e ",h_new[i+j*Nx+k*Nx*Ny]);
        }
        printf("\n");
      }
      printf("\n");
      printf("\n");
    }
    

    free(h_new);
    free(h_old);

}



