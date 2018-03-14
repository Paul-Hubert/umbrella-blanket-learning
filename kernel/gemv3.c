__kernel void gemv1(__global const float * a,__global const float * x,__global const float * b,
		    __global float * y, int ncol)
{
   int row = get_global_id(0);
   float sum = 0.0f;
   for(int col = 0; col<ncol; col++) {
      sum += a[row*ncol + col]*x[col];
   }
   y[row] = sum + b[row];
}