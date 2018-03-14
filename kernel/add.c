__kernel void add(__global const float * x,__global float * y)
{
   int col = get_global_id(0);
   y[col] += x[col];
}