__kernel void pointmul(__global const float * x1,__global const float * x2, __global float * y)
{
   int col = get_global_id(0);
   y[col] = x1[col] * x2[col];
}